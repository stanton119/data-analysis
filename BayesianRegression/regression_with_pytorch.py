# %% [markdown]
# # Regression with Pytorch
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

# independent features
n = 1000
m = 10
features = np.random.randn(n, m)

X = np.concatenate((np.ones((n, 1)), features), axis=1)
w = np.random.randn(m + 1, 1)
e = np.random.randn(n, 1)
y = X @ w + e
# %% [markdown]
# ## Statsmodels approach
# 
# With `statsmodels` we can apply the ordinary least squares solution to the above data to recover estimates of the weights, $w$.

# %% Linear regression with statsmodels
import statsmodels.api as sm

results = sm.OLS(y, X).fit()
print(results.summary())
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
import torch.optim as optim

X_t = torch.from_numpy(X)
y_t = torch.from_numpy(y)

w_t = torch.randn(m + 1, 1, dtype=torch.float64, requires_grad=True)
learning_rate = 5e-5
optimizer = optim.SGD([w_t], lr=learning_rate, momentum=0.0)
optimizer.zero_grad()

loss_values = []

for t in range(100):
    # Forward pass
    y_pred = X_t.mm(w_t)

    # Compute and print loss
    loss = (y_pred - y_t).pow(2).sum()
    loss_values.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

loss_values = np.array(loss_values)
# %% [markdown]
# The loss function from optimisation shows a consistent decrease as it converges:

# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
plt.style.use("seaborn-whitegrid")

plt.figure(num=None, figsize=(10, 6), dpi=80)
plt.plot(range(len(loss_values)), loss_values)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.savefig("images/loss.png")
plt.show()
# %% [markdown]
# ![](images/loss.png)
# 

# %% Results weights are equivalent to statsmodels
print(w)
print(w_t.flatten())
print(results.params.flatten())
print(results.summary())

plt.scatter(np.arange(len(w)), w)
plt.scatter(np.arange(len(w)), w_t.flatten().detach().numpy())
plt.scatter(np.arange(len(w)), results.params.flatten())



# %% [markdown]


# %% Create pytorch class

# %% Curveball
# %% New data set?


# 