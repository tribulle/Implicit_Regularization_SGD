import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import *

np.random.seed(42)

### Parameters
p = 128
n = 32
sigma2 = 2

epochs = 1000
intern_dim=10
optimizer='GD'
learning_rate = 0.123

t_max_reg = epochs

nb_avg = 1

model_name = 'SLN' #'MLP' or 'SLN'

### Repeating experiment
margin_gd = np.zeros((nb_avg, epochs))
margin_reg = np.zeros((nb_avg, epochs))
ws_gd_norm = np.zeros((nb_avg, epochs))
ws_reg_norm = np.zeros((nb_avg, epochs))

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
    w_reg = None # (epochs, p)
    ws_reg_norm = np.linalg.norm(w_reg, axis=-1)
    for j in range(epochs):
        margin_reg[i,j] = margin(data,observations,w_reg[j,:])

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
                             )

    margin_gd[i,:], ws_gd = train(model,
          input_Tensor,
          output_Tensor,
          lossFct = lambda x,y: exp_loss(x,y,true_param=torch.tensor(w_true)),
          optimizer=optimizer,
          epochs=epochs,
          batch_size=None,
          return_vals='margin',
          return_ws = True,
          init_norm = None,
          lr = learning_rate)
    ws_gd_norm[i,:] = np.linalg.norm(ws_gd, axis=-1)

mean_error = np.mean(margin_gd, axis=0)
mean_error_reg = np.mean(margin_reg, axis=0)
mean_norm = np.mean(ws_gd_norm, axis=0)
mean_norm_reg = np.mean(ws_reg_norm, axis=0)
nus = nu_classification(np.linspace(1e-1,t_max_reg, epochs))

fig,axs = plt.subplots(3,1)
axs[0].plot(nus, mean_norm, marker='*', markersize=5, label='GD')
axs[0].plot(nus, mean_norm_reg, marker='*', markersize=5, label='Regularized')
axs[0].set_title(f'Average $\|\theta\|_2$ over {nb_avg} experiments')
axs[0].set_xlabel(r'$\nu(t)$')
axs[0].set_ylabel(r'$\|\theta\|_2$')
axs[0].set_xscale('log')
axs[0].set_yscale('linear')
axs[0].grid(color='black', which="both", linestyle='-', linewidth=0.2)
axs[0].legend()

axs[1].plot(nus, margin_gd, marker='*', markersize=5, label='GD')
axs[1].plot(nus, margin_reg, marker='*', markersize=5, label='Regularized')
axs[1].set_title(f'Average margin over {nb_avg} experiments')
axs[1].set_xlabel(r'$\nu(t)$')
axs[1].set_ylabel(r'Margin')
axs[1].set_xscale('linear')
axs[1].set_yscale('linear')
axs[1].grid(color='black', which="both", linestyle='-', linewidth=0.2)
axs[1].legend()

axs[2].plot(range(epochs), np.abs(margin_gd-margin_reg), marker='*', markersize=5)
axs[2].set_title(f'Margin difference over {nb_avg} experiments')
axs[2].set_xlabel(r'Iteration $t$')
axs[2].set_ylabel(r'$|Margin(\theta_R(\nu(t))) - Margin(\theta_GD(t))|')
axs[2].set_xscale('log')
axs[2].set_yscale('linear')
axs[2].grid(color='black', which="both", linestyle='-', linewidth=0.2)

plt.grid(color='black', which="both", linestyle='-', linewidth=0.2)
plt.savefig(f'figures/classification_{model_name}_n{n}_t{epochs}_{optimizer}')