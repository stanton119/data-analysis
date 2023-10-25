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
*   Fitting a multimodal beta distribution with Pytorch - [notebook](neural_networks/fitting_multimodal_distributions/multimodal_beta_pytorch/multimodal_beta.ipynb)
*   Fitting a zero inflated Poisson distribution with Pytorch - [notebook](neural_networks/fitting_multimodal_distributions/zero_inflated_poisson_pytorch/zero_inflated_poisson.ipynb)
*   PyTorch: Linear regression to non linear probabilistic neural network - [notebook](neural_networks/nonlinear_regression/nonlinear_regression.ipynb)
*   TensorflowProbability: Linear regression to non linear probabilistic neural network - [notebook](neural_networks/tensorflow_probability/nonlinear_regression.ipynb)
*   Trying out `PyTorch Lightning` - [notebook](neural_networks/pytorch_lightning/pytorch_lightning_regression.ipynb)
*   Tensorflow - Do Neural Networks overfit?[notebook](neural_networks/overfitting_nn/overfitting_nn.ipynb)
*   Fitting a normal distribution with tensorflow probability - [notebook](neural_networks/tensorflow_probability/fit_gaussian_tfp.ipynb)
*   Binary loss functions - Is there a material difference between using `BCEWithLogitsLoss` and `CrossEntropyLoss` for binary classification tasks? - No - [notebook](neural_networks/binary_loss_functions.ipynb)
*   Does initialising the output of a neural net to match your target distribution help? - Yes - [notebook](neural_networks/output_layer_bias.ipynb)

### Recommenders
*   Exploring multi-armed bandit benchmarks - [notebook](recommenders/multi_armed_bandits_benchmarks/multi_armed_bandits.ipynb)

### Regression
*   Bootstrapping regression coefficients - Confirming theoretical regression coefficient distributions with bootstrapped samples - [notebook](regression/bootstrapped_regession_coefficients/bootstrap_regression.ipynb)
*   Interaction coefficients regularisation - [notebook](regression/interactions_linear_models/interactions_convergence.ipynb)
*   Sequential Bayesian linear regression model - [notebook](regression/sequential_bayesian_regression/sequential_bayesian_linear_regression.ipynb)
*   Bayesian regression adapting to non-stationary data - [notebook](regression/sequential_bayesian_regression/adaptive_coefficients.ipynb)
*   Binomial regression vs logistic regression - [notebook](regression/binomial_regression.ipynb)
*   Investigating double descent with linear regression - [notebook](regression/double_descent.ipynb)

### Time series
*   Speed of fitting and predict of `neuralprophet` vs `fbprophet` - [notebook](time_series/neural_prophet/neural_prophet_speed_test.ipynb)
*   Can we fit long AR models with `neuralprophet` - [notebook](time_series/neural_prophet/arima.ipynb)

### Tools/Python
*   Dask vs multiprocessing - Comparing the API of dask to multiprocessing for general functions - [python](tools_python/parallel_processing/dask_vs_multiprocessing.py)
*   Parquet datasets - Exporting writing dataframes to partitioned parquet files - [notebook](tools_python/parquet_datasets/parquet_datasets.ipynb)
*   Data generating functions from drawing data - [notebook](tools_python/data_generation_from_drawings.ipynb)

### Other
*   Analysis into European installed energy capacity - [notebook](other/energy_capacity/installed_energy_capacity.ipynb)
*   The Game of Life computed with convolution - [folder](other/game_of_life)
*   NBA - Analysis into LeBron James playing minutes - [notebook](other/nba/minutes_played/minutes_played.ipynb)
*   TFL - Analysis in to the number of bike trips taken per day in London - [notebook](other/tfl_cycles/data_exploration.ipynb)
*   NBA Score Trajectories - Flask app to show scores of a basketball match against time - [repo](https://github.com/stanton119/nba-scores)
*   NBA Shooting - Kedro data pipelines to plot player scoring probability distributions - [repo](https://github.com/stanton119/nba-analysis)
*   The classic birthday problem - [notebook](other/birthday_problems.ipynb)

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