# %% [markdown]
# # Bayesian Regression
# Todo:
# * Fit regression model with statsmodel
# * Fit regression model with pytorch
# * Fit bayesian regression model with pyro
# * Compare coefficient distribution to hessian approach
# * Repeat with tensorflow
#
# First we generate random data to test our models.
#
# We generate $$m$$ random features, $$X$$ from a Gaussian distribution.
# We append a column of ones to the front to act as our constant term.
# We then generate a set of random $$m+1$$ values to form our model weights.
#
# Then we form our output by creating a linear weighted sum of our features:
# $$y_i = \sum_m X_{i,m} * w_m + e_i$$
#
# We have added some Gaussian noise to create some uncertainty around of model estimates.

# %% generate regression data
import numpy as np
np.random.seed(0)
# independent features
n = 1000
m = 10
features = np.random.randn(n, m)
noise_std = 3.0

X = np.concatenate((np.ones((n, 1)), features), axis=1)
w = np.random.randn(m + 1, 1)
e = np.random.randn(n, 1) * noise_std
y = X @ w + e

# %% [markdown]
# ## Statsmodels approach
#
# With `statsmodels` we can apply the ordinary least squares solution to the above data to recover estimates of the weights, $w$.

# %% Linear regression with statsmodels
import statsmodels.api as sm

results = sm.OLS(y, X).fit()
w_sm = results.params.flatten()
print(results.summary())

y_est_sm = results.predict(X)


# %% [markdown]
# ```
# OLS Regression Results
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.912
# Model:                            OLS   Adj. R-squared:                  0.911
# Method:                 Least Squares   F-statistic:                     1023.
# Date:                Thu, 21 May 2020   Prob (F-statistic):               0.00
# Time:                        18:29:20   Log-Likelihood:                -1415.7
# No. Observations:                1000   AIC:                             2853.
# Df Residuals:                     989   BIC:                             2907.
# Df Model:                          10
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const         -0.2433      0.032     -7.612      0.000      -0.306      -0.181
# x1             1.0760      0.032     33.587      0.000       1.013       1.139
# x2            -1.1898      0.033    -36.496      0.000      -1.254      -1.126
# x3             0.1778      0.034      5.278      0.000       0.112       0.244
# x4             0.1600      0.033      4.911      0.000       0.096       0.224
# x5             0.3415      0.032     10.670      0.000       0.279       0.404
# x6             1.6577      0.031     54.026      0.000       1.598       1.718
# x7            -0.8801      0.031    -28.128      0.000      -0.942      -0.819
# x8             0.4310      0.032     13.368      0.000       0.368       0.494
# x9            -1.4959      0.033    -45.343      0.000      -1.561      -1.431
# x10           -1.0860      0.032    -33.522      0.000      -1.150      -1.022
# ==============================================================================
# Omnibus:                        0.774   Durbin-Watson:                   1.939
# Prob(Omnibus):                  0.679   Jarque-Bera (JB):                0.670
# Skew:                           0.055   Prob(JB):                        0.715
# Kurtosis:                       3.063   Cond. No.                         1.26
# ==============================================================================
#
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# ```

# %% [markdown]
# ## Pytorch approach
#
# We can solve the same linear regression problem in `pytorch`.
#
# The ordinary least squares method above minimise the negative likelihood function. In this case it is the MSE:
# $$loss = \sum_i (y-\hat{y})^2$$
# Minimising a function is a generic problem we can solve using the gradient descent method.
# This can be implemented by libraries such as pytorch.
#
# The problem is decomposed into a few steps:
# * Given an estimate of the model weights, we predict what our output, $$y$$, should be.
# * Calculate a loss function based on the difference between our prediction and the actual output. We use the above MSE function.
# * Update the model weights to improve the loss function.
# * Iterate the above until the weights have converged.
#
#
# We will apply this in pytorch.
# Pytorch has its own internal memory structures so we need to convert from our numpy arrays to torch tensors using `from_numpy()`.
# The weights estimates are initialised randomly. We require that the gradients are calculated for the weights so we use the `requires_grad` flag.
#
# We use the stochastic gradient descent optimiser to update the weights, the learning rate needs to be chosen appropriately.
#
# The forward step calculates the output of the network.
# The loss function is setup using pytorch functions which run over the tensor objects and allow the gradients to be calculated automatically.
#
# The gradients need to be reset each iteration as PyTorch accumulates the gradients on subsequent backward passes.
# The backwards step calculates the gradients automatically.
#
# The optimizer object then updates the model weights to minimise the loss function.
#
# We iterate over the data 100 times, at which point the weights have converged.

# %%
import torch
import time

torch.manual_seed(0)

X_t = torch.Tensor(X[:, 1:])
y_t = torch.Tensor(y)

model = torch.nn.Sequential(torch.nn.Linear(m, 1, bias=True))
loss_fn = torch.nn.MSELoss(reduction="sum")

learning_rate = 5e-5
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_values = []
for t in range(100):
    # Forward pass
    y_pred = model(X_t)

    # Compute and print loss
    loss = loss_fn(y_pred, y_t)
    loss_values.append(loss.item())

    # Update model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

loss_values = np.array(loss_values)

w_pt = list(model.parameters())
w_pt = np.concatenate(
    [w_pt[1][0].flatten().detach().numpy(), w_pt[0][0].flatten().detach().numpy()]
)
y_est_pt = model(X_t).detach().numpy().squeeze()

# %% [markdown]
# The loss function from optimisation shows a consistent decrease as it converges:
# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
plt.style.use("seaborn-whitegrid")

plt.figure(num=None, figsize=(10, 6), dpi=80)
plt.plot(range(len(loss_values)), loss_values)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Pytorch Regression")
plt.savefig("images/loss.png")
plt.show()
# %% [markdown]
# ![](images/loss.png)
#
# Looking at the model coefficients we can see that the neural network and the statsmodel linear regression give the same coefficient values:
# %% Results weights are equivalent to statsmodels
plt.figure(figsize=(10, 6))
plt.scatter(np.arange(len(w)), w, label="True", marker="o")
plt.scatter(
    np.arange(len(w)), w_pt, label="Pytorch", marker="x",
)
plt.scatter(
    np.arange(len(w)), w_sm, label="Statsmodels", marker="."
)
plt.xlabel("Coefficient number")
plt.ylabel("Value")
plt.title("Coefficient comparison")
plt.legend()
plt.savefig("images/coefficients.png")
plt.show()
# %% [markdown]
# ![](images/coefficients.png)







# %% Implementing in tensorflow

import tensorflow as tf

# Static model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            1, activation="linear", use_bias=True, input_shape=(m,)
        )
    ]
)

learning_rate = 3e-2
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.mse,
    metrics=[tf.keras.metrics.MSE],
)
# pytorch/statsmodels will optimise over the whole dataset at once, turn off batch size in tensorflow to match
history = model.fit(X[:, 1:], y, epochs=200, batch_size=y.shape[0])
plt.plot(history.history["loss"])
plt.show()

print(model.trainable_weights)

w_tf = np.concatenate(
    [
        model.trainable_weights[1].numpy(),
        model.trainable_weights[0].numpy().squeeze(),
    ]
)

y_est_tf = model.predict(X[:, 1:]).squeeze()


# %% Plot coefficients
fig, ax = plt.subplots(ncols=2, figsize=(10, 6))
ax[0].scatter(np.arange(len(w)), w, label="True", marker="o")
ax[0].scatter(
    np.arange(len(w)), w_pt, label="Pytorch", marker="x",
)
ax[0].scatter(
    np.arange(len(w)), w_sm, label="Statsmodels", marker="."
)
ax[0].scatter(np.arange(len(w)), w_tf, label="Tensorflow", marker="*")
plt.sca(ax[0])
plt.xlabel("Coefficient number")
plt.ylabel("Value")
plt.title("Coefficient comparison")
plt.legend()

ax[1].scatter(
    np.arange(len(w)), w_pt - w.squeeze(), label="Pytorch", marker="x",
)
ax[1].scatter(
    np.arange(len(w)),
    w_sm - w.squeeze(),
    label="Statsmodels",
    marker=".",
)
ax[1].scatter(
    np.arange(len(w)), w_tf - w.squeeze(), label="Tensorflow", marker="*"
)
plt.sca(ax[1])
plt.xlabel("Coefficient number")
plt.ylabel("Coefficent error")
plt.legend()

plt.savefig("images/coefficients_tf.png")
plt.show()

# %% Plot predictions
# Numerically close to statsmodels
plt.plot(y_est_sm, ".")
plt.plot(y_est_tf, ".")
plt.show()

plt.plot(y_est_tf - y_est_sm, ".")
plt.show()

np.mean((y_est_tf - y.squeeze()) ** 2)
np.mean((y_est_sm - y.squeeze()) ** 2)

# %% Bayesian model in tensorflow probability
import tensorflow_probability as tfp
tfd = tfp.distributions

# %% Part 1:
# Static model to mean only the mean of a gaussian output
# Generates gaussian distribution with the mean modelled by a single linear layer
# This should match the deterministic tensorflow graph above
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            1, activation="linear", use_bias=True, input_shape=(m,)
        ),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t,
                                scale=1)),
    ]
)

# Optimise log likelihood, finds log likelihood of our data given the distribution object returned from fitting
negloglik = lambda y, p_y: -p_y.log_prob(y)
learning_rate = 3e-2
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer,
    loss=negloglik,
    metrics=[tf.keras.metrics.MSE],
)
# pytorch/statsmodels will optimise over the whole dataset at once, turn off batch size in tensorflow to match
history = model.fit(X[:, 1:], y, epochs=200, batch_size=y.shape[0])
plt.plot(history.history["loss"])
plt.show()

w_tfp1 = np.concatenate(
    [
        model.trainable_weights[1].numpy(),
        model.trainable_weights[0].numpy().squeeze(),
    ]
)
# Model outputs are now distribution objects
y_est_tfp1 = model(X[:, 1:]).mean().numpy().squeeze()

# %% Part 2:
# Modelling aleatoric uncertainty
# Static model to model both the mean and std of y
# The std has to be positive so need a transform to ensure positive output
# The lambda function would take the 2 outputs from the dense layer and map 1 to mean, 1 to std
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            1 + 1, activation="linear", use_bias=True, input_shape=(m,)
        ),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[:,:1],
                                scale=1e-3 + tf.math.softplus(0.05 * t[...,1:]))),
    ]
)

# Optimise log likelihood, finds log likelihood of our data given the distribution object returned from fitting
negloglik = lambda y, p_y: -p_y.log_prob(y)
learning_rate = 5e-1
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer,
    loss=negloglik,
    metrics=[tf.keras.metrics.MSE],
)
# pytorch/statsmodels will optimise over the whole dataset at once, turn off batch size in tensorflow to match
history = model.fit(X[:, 1:], y, epochs=400, batch_size=y.shape[0])
plt.plot(history.history["loss"])
plt.show()

w_tfp2 = np.concatenate(
    [
        model.trainable_weights[1].numpy()[:1],
        model.trainable_weights[0].numpy()[:,0]
    ]
)

# Model outputs are now distribution objects, get dist parameters
y_est_tfp2 = model(X[:, 1:]).mean().numpy().squeeze()
y_est_tfp2_std = model(X[:, 1:]).stddev().numpy().squeeze()

# The weights for the mean are very similar to before, but now the standard deviation is modelled as well.
# The standard deviation in this case is 1, so the modelled standard deviation is not hugely useful.


# %%
fig, ax = plt.subplots(ncols=2, figsize=(10,6))
plt.sca(ax[0])
plt.plot(y_est_sm,'.', label='StatsModels')
plt.plot(y_est_pt,'.', label='Pytorch')
plt.plot(y_est_tf,'.', label='Tensorflow')
plt.plot(y_est_tfp1,'.', label='TensorflowProb')
plt.plot(y_est_tfp2,'.', label='TensorflowProbStd')
plt.legend()

plt.sca(ax[1])
plt.plot(y_est_sm - y.squeeze(),'.', label='StatsModels')
plt.plot(y_est_pt - y.squeeze(),'.', label='Pytorch')
plt.plot(y_est_tf - y.squeeze(),'.', label='Tensorflow')
plt.plot(y_est_tfp1 - y.squeeze(),'.', label='TensorflowProb')
plt.plot(y_est_tfp2 - y.squeeze(),'.', label='TensorflowProbStd')
plt.legend()

plt.show()

# %%
plt.plot(y_est_tfp2[:100],'.', label='TensorflowProbStd')
plt.plot(y_est_tfp2[:100] + y_est_tfp2_std[:100],'.', label='TensorflowProbStd')
plt.plot(y_est_tfp2[:100] - y_est_tfp2_std[:100],'.', label='TensorflowProbStd')
plt.show()

plt.plot(y_est_tfp2[:100], y_est_tfp2_std[:100],'.', label='TensorflowProbStd')
plt.show()

# %%
plt.plot(y_est_tf, y_est_tfp_mu, '.')
plt.show()
(y_est_tf - y.squeeze()) ** 2)
np.mean((y_est_sm 

# %% Curveball
# %% New data set?


# %% Quantile regression
# Static model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            1, activation="linear", use_bias=True, input_shape=(m,)
        )
    ]
)

# def quantile_loss(y_true, y_pred, quantile):
#     # Should avg to 0 if predictions are close to desired quantile
#     error = tf.subtract(tf.squeeze(y_true), tf.squeeze(y_pred))
#     loss = tf.reduce_mean(tf.maximum(quantile*error, (quantile-1)*error), axis=-1)
#     return loss
def quantile_loss(y_true, y_pred):
    # Should avg to 0 if predictions are close to desired quantile
    error = tf.subtract(tf.squeeze(y_true), tf.squeeze(y_pred))
    loss = tf.reduce_mean(tf.maximum(quantile*error, (quantile-1)*error), axis=-1)
    return loss

from functools import partial
quantiles = [0.1, 0.5, .9]
learning_rate = 3e-1
y_est_tf_q = {}
for quantile in quantiles:
    # loss_fnc = partial(quantile_loss, quantile=quantile)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=quantile_loss,
        metrics=[tf.keras.metrics.MSE],
    )
    # pytorch/statsmodels will optimise over the whole dataset at once, turn off batch size in tensorflow to match
    history = model.fit(X[:, 1:], y, epochs=200, batch_size=y.shape[0])
    y_est_tf_q[quantile] = model.predict(X[:, 1:]).squeeze()

y_est_tf_q.keys()
# %%
plt.plot(history.history["loss"])
plt.show()

print(model.trainable_weights)

w_tf_q50 = np.concatenate(
    [
        model.trainable_weights[1].numpy(),
        model.trainable_weights[0].numpy().squeeze(),
    ]
)
y_est_tf_q50 = model.predict(X[:, 1:]).squeeze()

plt.plot(y_est_tf_q[0.1], '.')
plt.plot(y_est_tf_q[0.5], '.')
plt.plot(y_est_tf_q[0.9], '.')
plt.show()

# %%
plt.plot(y_est_tfp2[:100],'.', label='TensorflowProbStd')
plt.plot(y_est_tfp2[:100] + y_est_tfp2_std[:100],'.', label='TensorflowProbStd')
plt.plot(y_est_tfp2[:100] - y_est_tfp2_std[:100],'.', label='TensorflowProbStd')
plt.show()

plt.plot(y_est_tf_q[0.1][:100],'.', label='TensorflowProbQ0.1')
plt.plot(y_est_tf_q[0.5][:100],'.', label='TensorflowProbQ0.5')
plt.plot(y_est_tf_q[0.9][:100],'.', label='TensorflowProbQ0.9')
plt.legend()
plt.show()

plt.plot(y_est_tfp2[:100],'.', label='TensorflowProbStd')
plt.plot(y_est_tf_q[0.5][:100],'.', label='TensorflowProbQ0.5')
plt.legend()
plt.show()


# %% Pyro
# http://pyro.ai/examples/bayesian_regression.html
#
# Need to specify a guide
# Multivariate gaussian to capture covariance between parameters

import pyro
from pyro.nn import PyroSample
from pyro.nn import PyroModule
from torch import nn

import os
from functools import partial
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist

pyro.set_rng_seed(1)


# Define forward pass model
class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(
            dist.Normal(0.0, 1.0)
            .expand([out_features, in_features])
            .to_event(2)
        )
        self.linear.bias = PyroSample(
            dist.Normal(0.0, 10.0).expand([out_features]).to_event(1)
        )

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0.0, 10.0))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean


from pyro.infer.autoguide import (
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    init_to_mean,
)

model = BayesianRegression(m, 1)
# Incorrectly assumes a diagonal covariance matrix
guide = AutoDiagonalNormal(model)
guide = AutoMultivariateNormal(model, init_loc_fn=init_to_mean)


from pyro.infer import SVI, Trace_ELBO


adam = pyro.optim.Adam({"lr": 0.03})
svi = SVI(model, guide, adam, loss=Trace_ELBO())

pyro.clear_param_store()
loss = []
for j in range(2000):
    # calculate the loss and take a gradient step
    loss.append(svi.step(X_t, y_t))
    if j % 100 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss[-1] / len(X_t)))
loss = np.array(loss)

# %%
plt.plot(loss / len(X_t))
plt.show()
X_t.shape


from pyro.infer import Predictive

predictive = Predictive(
    model,
    guide=guide,
    num_samples=800,
    return_sites=("linear.weight", "obs", "_RETURN"),
)
samples = predictive(X_t)
samples.keys()
y_est = samples["obs"].mean(axis=0)

y_est = model(X_t)
plt.plot(y_est.detach().numpy(), ".")
plt.plot(y_t, ".")
y_t
plt.show()

# %%
# Size of model params = m + 1 for sigma + 1 for bias = m+2
guide.requires_grad_(False)

for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name), pyro.param(name).shape)

guide.quantiles([0.25, 0.5, 0.75])


w_t
