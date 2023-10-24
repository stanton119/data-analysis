# Read me

Repo directory:  
*   Projects are split by folders

## Topic areas
### Causal inference
*   Causal regression - [notebook](causal_inference/causal_regression_coefficient.ipynb)
*   Causal regression with DoWhy - [notebook](causal_inference/causal_regression_coefficient_dowhy.ipynb)
*   Double machine learning and marginal effects - [notebook](causal_inference/doubleml_marginal_effects.ipynb)

### Machine vision
*   Using Google's `mediapipe` to try simulate a 3D screen - [folder](machine_vision/3d_screen)
*   Using Google's `mediapipe`, measure the distance of a face to the screen from a webcam feed - [folder](machine_vision/face_distance)
*   FashionCNN - Convolution neural network for predicting the Fashion MNIST dataset - [notebook](machine_vision/fashion_cnn/fashion_batch_norm.ipynb)
*   FashionCNN - Batch normalisation layer applied to the above CNN model - [notebook](machine_vision/fashion_cnn/fashion_batch_norm.ipynb)

### Neural networks
*   Autoencoders - Using PCA to compress MNIST images - [notebook](neural_networks/autoencoders/pca.ipynb)
*   Autoencoders - Using a dense autoencoder to compress MNIST images - [notebook](neural_networks/autoencoders/autoencoder.ipynb)
*   Implementing an elastic net model in PyTorch - [notebook](neural_networks/elastic_net/elastic_linear.ipynb)
*   Fitting distributions with variational inference - Simple example fitting a Gaussian distribution to data with Pyro - [notebook](neural_networks/fit_dist_with_pyro/fit_pyro_distribution.ipynb)
*   Fitting distributions with variational inference - Simple example fitting a beta distribution to data with Pyro - [notebook](neural_networks/fit_dist_with_pyro/fit_beta_distribution.ipynb)


# Appendix
## Model frameworks
### [Pytorch](PyTorchStuff/)
*   [PyTorchStuff/nonlinear_regression/nonlinear_regression.md](https://github.com/stanton119/data-analysis/blob/master/PyTorchStuff/nonlinear_regression/nonlinear_regression.md)
    *   Linear regression to non linear probabilistic neural network.
*   [PyTorchStuff/pytorch_lightning/pytorch_lightning_regression.md](https://github.com/stanton119/data-analysis/blob/master/PyTorchStuff/pytorch_lightning/pytorch_lightning_regression.md)
    *   Trying out `PyTorch Lightning`

*   [FittingMultimodalDistributions/multimodal_beta_pytorch/multimodal_beta.md](https://github.com/stanton119/data-analysis/blob/master/FittingMultimodalDistributions/multimodal_beta_pytorch/multimodal_beta.md)
    *   Fitting a multimodal beta distribution via gradient descent
*   [FittingMultimodalDistributions/zero_inflated_poisson_pytorch/zero_inflated_poisson.md](https://github.com/stanton119/data-analysis/blob/master/FittingMultimodalDistributions/zero_inflated_poisson_pytorch/zero_inflated_poisson.md)
    *   Fitting a zero-inflated Poisson distribution via gradient descent
*   [PyTorchStuff/binary_loss_functions.ipynb](https://github.com/stanton119/data-analysis/blob/master/PyTorchStuff/binary_loss_functions.ipynb)
    *   Is there a material difference between using `BCEWithLogitsLoss` and `CrossEntropyLoss` for binary classification tasks? - No
*   [PyTorchStuff/output_layer_bias.ipynb](https://github.com/stanton119/data-analysis/blob/master/PyTorchStuff/output_layer_bias.ipynb)
    *   Does initialising the output of a neural net to match your target distribution help? - Yes

### [Pyro](FitDistWithPyro/)


### [Tensorflow Probability](TensorflowProbability/)
*   [TensorflowProbability/nonlinear_regression.ipynb](https://github.com/stanton119/data-analysis/blob/master/TensorflowProbability/nonlinear_regression.ipynb)
    *   Linear regression to non linear probabilistic neural network.
*   [TensorflowProbability/fit_gaussian_tfp.ipynb](https://github.com/stanton119/data-analysis/blob/master/TensorflowProbability/fit_gaussian_tfp.ipynb)
    *   Fitting a normal distribution with tensorflow probability.

### [Tensorflow](TensorflowStuff/)
*   [TensorflowStuff/overfitting_nn.ipynb](https://github.com/stanton119/data-analysis/blob/master/TensorflowStuff/overfitting_nn.ipynb)
    *   Do Neural Networks overfit?

### Others
*   [TimeSeries/neural_prophet/neural_prophet_speed_test.md](https://github.com/stanton119/data-analysis/blob/master/TimeSeries/neural_prophet/neural_prophet_speed_test.md)
    *   Speed of fitting and predict of `neuralprophet` vs `fbprophet`
*   [TimeSeries/neural_prophet/arima.md](https://github.com/stanton119/data-analysis/blob/master/TimeSeries/neural_prophet/arima.md)
    *   Can we fit long AR models with `neuralprophet`

## Techniques
*   [ParquetDatasets/parquet_datasets.ipynb](https://github.com/stanton119/data-analysis/blob/master/ParquetDatasets/parquet_datasets.ipynb)
    *   Exporting dataframes to partitioned parquet files.
*   [Dask vs multiprocessing](https://github.com/stanton119/data-analysis/blob/master/parallel_processing/dask_vs_multiprocessing.py)
    *   Comparing the API fo dask to multiprocessing for general functions
*   [recommenders/multi_armed_bandits_benchmarks](https://github.com/stanton119/data-analysis/blob/master/recommenders/multi_armed_bandits_benchmarks/multi_armed_bandits.md)
    *   Exploring multi-armed bandit benchmarks

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
*   Computer vision
*   NLP
    *   Keyword extraction from reviews etc.
    *   Sentiment analysis
*   Gaussian processes
*   Bayesian regression
*   Recommender systems
    *   Automatic playlist continuation
    *   Thompson sampling example
*   Switch away from Matplotlib, try plotly express in markdown
*   Quantile regression in pytorch
    *   Lasso regression
    *   Dropout better than regularisation?
*   Data engineering
    *   Polars
    *   DuckDB
*   Docker

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
    *   NN to predict tab from music sections
*   Word embeddings plot with hiplot
    *   Plot with PCA first and compare with hiplot
*   Compare linear regression MC dropout to theoretical results
*   Optimal car charging schedule based on energy prices or carbon output
*   Media pipe - 3d audio
    *   Face distance javascript web app with react
*   Covid UK plot against time on a map
    *   https://www.reddit.com/r/dataisbeautiful/comments/pay78n/oc_active_covid19_cases_per_capita_in_usa_1212020/
*   Autoencoder using transfer learning?
    *   what do we use for the decoder?
    *   MNIST auto-encoder to digit classifier
*   Fit a sinusoid to noisy data
    *   Fourier
    *   Gradient descent
    *   MCMC
    *   Variational inference
*   Double dip loss trajectories
*   Fitting NNs to common functions (exp etc.), deep vs wide, number of parameters for given error
*   Fit a NN to seasonal data with fourier series components
*   Causal inference
    *   DoubleML on heart data to find CATE
    *   DoubleML on dummy data vs other causal models. How robust are they to model mis-specification and missing confounders?
    *   Inverse propensity scoring - comparing different methods - manual Inverse Probability of Treatment Weighting, as variance in regression, sample weights, econML based. Do they match?
*   Github action to publish ipynbs to markdown
*   Hierarchical models
    *   Mixed effects model - is it the same as a fixed effects model (lin/log regression) with one hot encoding for the categorical variables + a fixed effect?
    *   Hierarchical bayesian models - for when we have categorical features with share effects over other features
    *   Fit with MCMC
    *   Similarities to ridge regression - only some coefficients are regularised
    *   Generate data and fit each model
    *   Ref
        *   https://www.youtube.com/watch?v=38yOWMMCeMk&list=WL&index=5
*   Bimomial regression = logistic regression
*   Linear regression = logistic regression, relationship to Linear Thompson Sampling
*   Blurred images classifier
    *   ImageNet based, data augment to blur images.
*   Country embeddings - create country embeddings by predicting various macro level metrics (GDP, population etc. in a multi task model), from OHE through a NN. Does the embedding space make sense?
*   [MovieLens](https://grouplens.org/datasets/movielens/) dataset to get title embeddings, find nearest neighbour titles
    *   Using word2vec to predict similar titles. Train on movies watched. Similar given as titles streamed by the same customer
*   Finding similar images in a photo library - given a few examples find similar photos
    *   Use an image net model. Find new example images, positive and negative. Fine tune the model via a classification task. Predict prob of positive result for unseen images. Use the latent space embeddings to find cosine similarity between images.
    *   Build small image dataset from cifar 10. Compare models - PCA/logistic regression, CNN, efficientNet, transfer learnt weights
    *   Build lookup table of image and its compact embedding. Given a new image find the inner product with the other images
*   Fourier transform via linear regression on sinusoids. Similar approach with Lasso regression to find compressed sensing approaches, with non-uniform sampling.
* Multi task neural network training
    * train a single model to predict multiple ready fields from a single dataset
* Learning interactions after main effects
    * Create a synthetic dataset with interaction effects
    * Fit models with/without interaction effects with incrementally more data
    * Does interaction effects model fit with more data to higher out of sample performance
    * Fit a model without interactions, introduction interactions after in some way. Perhaps tight priors on an interaction effects more which are relaxed with time.
* A/B test distribution comparison
    * We often compare just the means. If we find plot a Q-Q plot is it more informative, bootstrapping would construct confidence intervals

## TODO
* Restructure:
    * Remove .md/images, keep notebooks only where possible
    * Change readme links
    * Optional action to create markdowns on separate branch
* rename environment/requirements files to match the notebook
* change markdown actions to stop markdown conversion
* update blog articles for markdown images
* Add year to each analysis link