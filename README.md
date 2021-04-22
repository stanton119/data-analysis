# Read me

Repo directory:  
* Projects are split by folders

## Applied datasets
* [TFL Cycle Analysis](https://github.com/stanton119/data-analysis/tree/master/TFLCycles)
  * Analysis in to the number of bike trips taken per day in London.
* [NBA Score Trajectories](https://github.com/stanton119/nba-scores)
  * Flask app to show scores of a basketball match against time.
* [Installed energy capacity](https://github.com/stanton119/data-analysis/blob/master/EnergyCapacity/installed_energy_capacity.ipynb)
  * Analysis into European installed energy capacity
* [NBA LeBron Minutes](https://github.com/stanton119/data-analysis/tree/master/NBA/minutes_played/minutes_played.md)
  * Analysis into LeBron James playing minutes
* [NBA Shooting Data](https://github.com/stanton119/data-analysis/tree/master/NBA/NBAShotSelection)
  * Not finished yet - animations of shots taken by Kobe Bryant over time.

## Model frameworks
### [Pytorch](PyTorchStuff/)
* [PyTorchStuff/nonlinear_regression/nonlinear_regression.md](https://github.com/stanton119/data-analysis/blob/master/PyTorchStuff/nonlinear_regression/nonlinear_regression.md)
  * Linear regression to non linear probabilistic neural network.
* [PyTorchStuff/pytorch_lightning/pytorch_lightning_regression.md](https://github.com/stanton119/data-analysis/blob/master/PyTorchStuff/pytorch_lightning/pytorch_lightning_regression.md)
  * Trying out `PyTorch Lightning`

### [Pyro](FitDistWithPyro/)
* [FitDistWithPyro/fit_pyro_distribution.ipynb](https://github.com/stanton119/data-analysis/blob/master/FitDistWithPyro/fit_pyro_distribution.ipynb)
  * Simple example fitting a Gaussian distribution to data using Pyro.
* [FitDistWithPyro/fit_beta_distribution.ipynb](https://github.com/stanton119/data-analysis/blob/master/FitDistWithPyro/fit_beta_distribution.ipynb)
  * Simple example fitting a beta distribution to data using Pyro.

### [Tensorflow Probability](TensorflowProbability/)
* [TensorflowProbability/nonlinear_regression.ipynb](https://github.com/stanton119/data-analysis/blob/master/TensorflowProbability/nonlinear_regression.ipynb)
  * Linear regression to non linear probabilistic neural network.
* [TensorflowProbability/fit_gaussian_tfp.ipynb](https://github.com/stanton119/data-analysis/blob/master/TensorflowProbability/fit_gaussian_tfp.ipynb)
  * Fitting a normal distribution with tensorflow probability.

### [Tensorflow](TensorflowStuff/)
* [TensorflowStuff/overfitting_nn.ipynb](https://github.com/stanton119/data-analysis/blob/master/TensorflowStuff/overfitting_nn.ipynb)
  * Do Neural Networks overfit?
* [FashionCNN/fashion_vae.ipynb](https://github.com/stanton119/data-analysis/blob/master/FashionCNN/fashion_vae.ipynb)
  * Convolution neural network for predicting the Fashion MNIST dataset.
* [FashionCNN/fashion_batch_norm.ipynb](https://github.com/stanton119/data-analysis/blob/master/FashionCNN/fashion_batch_norm.ipynb)
  * Batch normalisation layer applied to the above CNN model

### Others
* [TimeSeries/neural_prophet/neural_prophet_speed_test.md](https://github.com/stanton119/data-analysis/blob/master/TimeSeries/neural_prophet/neural_prophet_speed_test.md)
* [TimeSeries/neural_prophet/arima.md](https://github.com/stanton119/data-analysis/blob/master/TimeSeries/neural_prophet/arima.md)

## Techniques
* [ParquetDatasets/parquet_datasets.ipynb](https://github.com/stanton119/data-analysis/blob/master/ParquetDatasets/parquet_datasets.ipynb)
  * Exporting dataframes to partitioned parquet files.

* [BootstrappedRegessionCoefficients/bootstrap_regression.ipynb](https://github.com/stanton119/data-analysis/blob/master/BootstrappedRegessionCoefficients/bootstrap_regression.ipynb)
  * Confirming theoretical regression coefficient distributions with bootstrapped samples.



## Installation
The various analysis was built in Python 3.

### Virtual environment setup
Some projects have their own requirements/environment. The general setup is installed by:

```
python3 -m venv dataAnalysisEnv
source dataAnalysisEnv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Standard library
Custom library installed as a dev library for continued development

### VSCode
Use the `settings.json` file in the repo

## Future aims
* Time series forecasting
* Pytorch
* Causal inference
* Data validations - great expectations
* PySpark
* NLP
* Computer vision
* Gaussian processes
* Bayesian regression
* Tensorflow/pytorch - 1D functions
* FashionMNIST VAE
* Switch away from Matplotlib
* Project work
  * Optimal car charging schedule
  * NBA player VAE - how are players related

## Questions
* Do VAEs work well with (linearly) correlated features
  * Can they form orthogonal features?