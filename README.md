# Implicit_Regularization_SGD
Project for Optimization for ML from EPFL.

We investigate the implicit regularization of SGD in non-linear least square setting.

## Our work
This study has 3 parts:
- Reproduction of the results of REF
- Extension of the results to fully connected linear neural networks
- Test on a real-world dataset
  
## Code
The results are generated by a `crossvalidation` file, to fine-tune the hyperparameters of the models.
Then a `data_generation` will train several models. Finally, the `study` files will average and plot the results.

The two first points of our work can be executed by running the `main.py` file, the different settings are commented in the file. The default mode is to reproduce the results of the article.
To study the real-world dataset, one must run individually the `benefits_real_data_set_crossvalidation.py`, then `benefits_real_data_set_data_generation.py` and finally `benefits_study_real_dataset.py`.
