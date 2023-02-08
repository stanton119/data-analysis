# Read me

Repo directory:  
*   Projects are split by folders

## Model frameworks
### [Pytorch](PyTorchStuff/)
*   [PyTorchStuff/nonlinear_regression/nonlinear_regression.md](https://github.com/stanton119/data-analysis/blob/master/PyTorchStuff/nonlinear_regression/nonlinear_regression.md)
    *   Linear regression to non linear probabilistic neural network.
*   [PyTorchStuff/pytorch_lightning/pytorch_lightning_regression.md](https://github.com/stanton119/data-analysis/blob/master/PyTorchStuff/pytorch_lightning/pytorch_lightning_regression.md)
    *   Trying out `PyTorch Lightning`
*   [PyTorchStuff/elastic_net/elastic_linear.md](https://github.com/stanton119/data-analysis/blob/master/PyTorchStuff/elastic_net/elastic_linear.md)
    *   Implementing an elastic net model in PyTorch
*   [FittingMultimodalDistributions/multimodal_beta_pytorch/multimodal_beta.md](https://github.com/stanton119/data-analysis/blob/master/FittingMultimodalDistributions/multimodal_beta_pytorch/multimodal_beta.md)
    *   Fitting a multimodal beta distribution via gradient descent
*   [FittingMultimodalDistributions/zero_inflated_poisson_pytorch/zero_inflated_poisson.md](https://github.com/stanton119/data-analysis/blob/master/FittingMultimodalDistributions/zero_inflated_poisson_pytorch/zero_inflated_poisson.md)
    *   Fitting a zero-inflated Poisson distribution via gradient descent

### [Pyro](FitDistWithPyro/)
*   [FitDistWithPyro/fit_pyro_distribution.ipynb](https://github.com/stanton119/data-analysis/blob/master/FitDistWithPyro/fit_pyro_distribution.ipynb)
    *   Simple example fitting a Gaussian distribution to data using Pyro.
*   [FitDistWithPyro/fit_beta_distribution.ipynb](https://github.com/stanton119/data-analysis/blob/master/FitDistWithPyro/fit_beta_distribution.ipynb)
    *   Simple example fitting a beta distribution to data using Pyro.

### [Tensorflow Probability](TensorflowProbability/)
*   [TensorflowProbability/nonlinear_regression.ipynb](https://github.com/stanton119/data-analysis/blob/master/TensorflowProbability/nonlinear_regression.ipynb)
    *   Linear regression to non linear probabilistic neural network.
*   [TensorflowProbability/fit_gaussian_tfp.ipynb](https://github.com/stanton119/data-analysis/blob/master/TensorflowProbability/fit_gaussian_tfp.ipynb)
    *   Fitting a normal distribution with tensorflow probability.

### [Tensorflow](TensorflowStuff/)
*   [TensorflowStuff/overfitting_nn.ipynb](https://github.com/stanton119/data-analysis/blob/master/TensorflowStuff/overfitting_nn.ipynb)
    *   Do Neural Networks overfit?
*   [FashionCNN/fashion_cnn.ipynb](https://github.com/stanton119/data-analysis/blob/master/FashionCNN/fashion_cnn.ipynb)
    *   Convolution neural network for predicting the Fashion MNIST dataset.
*   [FashionCNN/fashion_batch_norm.ipynb](https://github.com/stanton119/data-analysis/blob/master/FashionCNN/fashion_batch_norm.ipynb)
    *   Batch normalisation layer applied to the above CNN model

### Others
*   [TimeSeries/neural_prophet/neural_prophet_speed_test.md](https://github.com/stanton119/data-analysis/blob/master/TimeSeries/neural_prophet/neural_prophet_speed_test.md)
    *   Speed of fitting and predict of `neuralprophet` vs `fbprophet`
*   [TimeSeries/neural_prophet/arima.md](https://github.com/stanton119/data-analysis/blob/master/TimeSeries/neural_prophet/arima.md)
    *   Can we fit long AR models with `neuralprophet`
*   [machine_vision/3d_screen](https://github.com/stanton119/data-analysis/tree/master/machine_vision/3d_screen)
    *   Using Google's `mediapipe` to try simulate a 3D screen
*   [machine_vision/face_distance](https://github.com/stanton119/data-analysis/tree/master/machine_vision/face_distance)
    *   Using Google's `mediapipe`, measure the distance of a face to the screen from a webcam feed

## Techniques
*   [ParquetDatasets/parquet_datasets.ipynb](https://github.com/stanton119/data-analysis/blob/master/ParquetDatasets/parquet_datasets.ipynb)
    *   Exporting dataframes to partitioned parquet files.
*   [Dask vs multiprocessing](https://github.com/stanton119/data-analysis/blob/master/parallel_processing/dask_vs_multiprocessing.py)
    *   Comparing the API fo dask to multiprocessing for general functions
*   [recommenders/multi_armed_bandits_benchmarks](https://github.com/stanton119/data-analysis/blob/master/recommenders/multi_armed_bandits_benchmarks/multi_armed_bandits.md)
    *   Exploring multi-armed bandit benchmarks
*   [PCA image compression](https://github.com/stanton119/data-analysis/blob/master/PyTorchStuff/autoencoders/pca.md)
    *   Using PCA to compress MNIST images
*   [Dense autoencoder image compression](https://github.com/stanton119/data-analysis/blob/master/PyTorchStuff/autoencoders/autoencoder.md)
    *   Using a dense autoencoder to compress MNIST images

### Bayesian Regression
*   [BootstrappedRegessionCoefficients/bootstrap_regression.ipynb](https://github.com/stanton119/data-analysis/blob/master/BootstrappedRegessionCoefficients/bootstrap_regression.ipynb)
    *   Confirming theoretical regression coefficient distributions with bootstrapped samples.
*   [SequentialBayesianRegression/sequential_bayesian_linear_regression.md](https://github.com/stanton119/data-analysis/blob/master/SequentialBayesianRegression/sequential_bayesian_linear_regression.md)
    *   Sequential Bayesian linear regression model
*   [SequentialBayesianRegression/adaptive_coefficients.md](https://github.com/stanton119/data-analysis/blob/master/SequentialBayesianRegression/adaptive_coefficients.md)
    *   Bayesian regression adapting to non-stationary data

### Causal Inference
*   [CausalInference/causal_regression_coefficient.ipynb](https://github.com/stanton119/data-analysis/blob/master/CausalInference/causal_regression_coefficient.ipynb)
    *   Exploring the effect of confounding variables on regression coefficients

## Applied datasets
*   [TFL Cycle Analysis](https://github.com/stanton119/data-analysis/tree/master/TFLCycles)
    *   Analysis in to the number of bike trips taken per day in London.
*   [NBA Score Trajectories](https://github.com/stanton119/nba-scores)
    *   Flask app to show scores of a basketball match against time.
*   [Installed energy capacity](https://github.com/stanton119/data-analysis/blob/master/EnergyCapacity/installed_energy_capacity.ipynb)
    *   Analysis into European installed energy capacity
*   [NBA LeBron Minutes](https://github.com/stanton119/data-analysis/tree/master/NBA/minutes_played/minutes_played.md)
    *   Analysis into LeBron James playing minutes
*   [NBA Shooting - Kedro project](https://github.com/stanton119/data-analysis/tree/master/NBA/nba-analysis)
    *   Kedro data pipelines to plot player scoring probability distributions
*   [NBA Shooting Data](https://github.com/stanton119/data-analysis/tree/master/NBA/NBAShotSelection)
    *   Not finished yet - animations of shots taken by Kobe Bryant over time.

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

### Markdown from Notebooks
```
jupyter nbconvert notebook.ipynb --to markdown
```
This is automated via github actions.

### Standard library
Custom library installed as a dev library for continued development

### VSCode
Use the `settings.json` file in the repo

## Future areas

Aim is for future work to be incorporated by working on separate branches and merge to master when finished.

### Tools/areas to explore
*   Time series forecasting
    *   Greykite
        *   https://arxiv.org/abs/2105.01098
        *   https://towardsdatascience.com/linkedins-response-to-prophet-silverkite-and-greykite-4fd0131f64cb
        *   Imputation of missing regressors
        *   Change points in seasonalities
        *   Quantiles loss
        *   Utilities for diagnosing
        *   faster inference
        *   Autoregressive
    *   LightGBM
    *   Orbit
        *   https://eng.uber.com/orbit/
*   Deep learning
    *   Pytorch
    *   Embeddings
    *   Tensorflow/pytorch - 1D functions
    *   FashionMNIST VAE
*   Causal inference
*   Data validations - great expectations
    *   https://github.com/tamsanh/kedro-great
*   Docker
*   Computer vision
*   NLP
    *   Keyword extraction from reviews etc.
    *   Sentiment analysis
*   Gaussian processes
*   Bayesian regression
*   Recommender systems
    *   Automatic playlist continuation
    *   Thompson sampling example
*   Switch away from Matplotlib
*   Quantile regression in pytorch
    *   Lasso regression
    *   Dropout better than regularisation?
*   GBM/LightGBM

### Datasets to explore
*   https://archive-beta.ics.uci.edu/ml/datasets#music
*   https://ritual.uh.edu/mpst-2018/
    *   NLP


### Tasks
*   Build project template repo
*   Publish interpret-ml piece
*   NBA
    *   Player position classification model
    *   Bayesian sequential team rating
    *   Player VAE - how are players related
        *   College stats to NBA VAE
*   M5/M4 forecasting
    *   Walmart demand forecasting
    *   with LightGBM
    *   Greykite
*   PCA via embedding layer
*   NN to predict tempo from song, generate dummy dataset
*   Word embeddings plot with hiplot
    *   Plot with PCA first and compare with hiplot
*   Compare linear regression MC dropout to theoretical results
*   Optimal car charging schedule
*   Media pipe - 3d audio
    *   Face distance javascript web app with react
*   Covid UK plot against time on a map
    *   https://www.reddit.com/r/dataisbeautiful/comments/pay78n/oc_active_covid19_cases_per_capita_in_usa_1212020/
*   Autoencoder using transfer learning?
    *   what do we use for the decoder?
*   Fit a sinusoid to noisy data
    *   Fourier
    *   Gradient descent
    *   MCMC
    *   Variational inference
*   Double dip loss trajectories
*   Fitting NNs to common functions (exp etc.), deep vs wide, number of parameters for given error
*   Fit a NN to seasonal data with fourier series components
*   DoubleML on heart data to find CATE
*   Github action to publish ipynbs to markdown
*   Mixed effects model - is it the same as a fixed effects model (lin/log regression) with one hot encoding for the categorical variables + a fixed effect?
*   Bimomial regression = logistic regression
*   Linear regression = logistic regression, relationship to Linear Thompson Sampling