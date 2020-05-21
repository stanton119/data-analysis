# %%
import pandas as pd

with open('table.html') as f:
    df = pd.read_html(
        f, decimal=',', flavor='bs4'
    )[0]
# %%
df.columns
temp = df.loc[(df['Preis in € (ca.)']<150) & (df['Sicherheit  (Gewichtung 50%)']<=20), :]
temp.sort_values(['Preis in € (ca.)'])


# %%
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

# %%
X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X, y)
gpr.score(X, y)

gpr.predict(X[:2, :], return_std=True)

# %%
import matplotlib.pyplot as plt

plt.hist(y)

import seaborn as sb
import pandas as pd

df = pd.DataFrame(data=X)
sb.pairplot(df)
plt.show()

plt.scatter(X[:, 0], y)

y_est = gpr.predict(X)

plt.scatter(y, y_est)

# %%

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(1)


def f(x):
    """The function to predict."""
    return x * np.sin(x)


# %% ----------------------------------------------------------------------
#  First the noiseless case
X = np.atleast_2d([1.0, 3.0, 5.0, 6.0, 7.0, 8.0]).T

# Observations
y = f(X).flatten()

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0, 10, 1000)).T

# Instantiate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.plot(x, f(x), "r:", label=r"$f(x) = x\,\sin(x)$")
plt.plot(X, y, "r.", markersize=10, label="Observations")
plt.plot(x, y_pred, "b-", label="Prediction")
plt.fill(
    np.concatenate([x, x[::-1]]),
    np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]),
    alpha=0.5,
    fc="b",
    ec="None",
    label="95% confidence interval",
)
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.ylim(-10, 20)
plt.legend(loc="upper left")

# %% ----------------------------------------------------------------------
# now the noisy case
X = np.linspace(0.1, 9.9, 20)
X = np.atleast_2d(X).T

# Observations and noise
y = f(X).ravel()
dy = 0.5 + 1.0 * np.random.random(y.shape)
noise = np.random.normal(0, dy)
y += noise

# Instantiate a Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel, alpha=dy ** 2, n_restarts_optimizer=10)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.plot(x, f(x), "r:", label=r"$f(x) = x\,\sin(x)$")
plt.errorbar(X.ravel(), y, dy, fmt="r.", markersize=10, label="Observations")
plt.plot(x, y_pred, "b-", label="Prediction")
plt.fill(
    np.concatenate([x, x[::-1]]),
    np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]),
    alpha=0.5,
    fc="b",
    ec="None",
    label="95% confidence interval",
)
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.ylim(-10, 20)
plt.legend(loc="upper left")

plt.show()

# %%

# %%
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 1000)

# %%
plt.plot(x, np.sign(x) * np.abs(x) ** 3)
plt.plot(x, np.abs(x) ** 2)
plt.show()

# %%
