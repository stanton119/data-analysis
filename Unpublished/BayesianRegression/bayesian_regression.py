# %% [markdown]
# # Transport for London Cycle Data Exploration
#
# ## Dataset
# The data was provided from TFL and was retrieved from Kaggle:  
# https://www.kaggle.com/hmavrodiev/london-bike-sharing-dataset  
# The dataset counts the number of journeys made per hour in each day of 2015-2017.
# There are 17414 rows.

# %%
import data_proc as dp


# %% [markdown]
# |    | timestamp           |   cnt |   t1 |   t2 |   hum |   wind_speed |   weather_code |   is_holiday |   is_weekend |   season |
# |---:|:--------------------|------:|-----:|-----:|------:|-------------:|---------------:|-------------:|-------------:|---------:|
# |  0 | 2015-01-04 00:00:00 |   182 |  3   |  2   |  93   |          6   |              3 |            0 |            1 |        3 |
# |  1 | 2015-01-04 01:00:


# %% [markdown]
# # Regression with Pytorch
#asdf

# %% generate data
import numpy as np

# independent features
n = 1000
m = 10
features = np.random.randn(n, m)

X = np.concatenate((np.ones((n, 1)), features), axis=1)
w = np.random.randn(m + 1, 1)
e = np.random.randn(n, 1)
y = X @ w + e


# %% Linear regression
import statsmodels.api as sm

results = sm.OLS(y, X).fit()
print(results.summary())

# %% Solving with pytorch
# No need to normalise features here
import torch

# Implement with pytorch autograd
X_t = torch.from_numpy(X)
y_t = torch.from_numpy(y)
w_t = torch.randn(m + 1, 1, dtype=torch.float64, requires_grad=True)

learning_rate = 1e-5
for t in range(500):
    # Forward pass
    y_pred = X_t.mm(w_t)

    # Compute and print loss
    loss = (y_pred - y_t).pow(2).sum()
    print(t, loss.item())

    loss.backward()

    # You can also use torch.optim.SGD to achieve this.
    with torch.no_grad():
        w_t -= learning_rate * w_t.grad

        # Manually zero the gradients after updating weights
        w_t.grad.zero_()

# %% Using SGD optimiser
import torch
import torch.optim as optim

X_t = torch.from_numpy(X)
y_t = torch.from_numpy(y)

w_t = torch.randn(m + 1, 1, dtype=torch.float64, requires_grad=True)
optimizer = optim.SGD([w_t], lr=learning_rate, momentum=0.9)
optimizer.zero_grad()
for t in range(500):
    # Forward pass
    y_pred = X_t.mm(w_t)

    # Compute and print loss
    loss = (y_pred - y_t).pow(2).sum()
    print(t, loss.item())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# %% Results weights are equivalent to statsmodels
w
w_t.flatten()
results.params.flatten()
print(results.summary())

# %% Bayesian network
# Reference: http://pyro.ai/examples/bayesian_regression.html