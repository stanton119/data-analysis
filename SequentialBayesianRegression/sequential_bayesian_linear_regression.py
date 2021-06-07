# %% [markdown]
# # Bayesian Linear Regression
# Reformulating linear regression to give epistemic uncertainty.
# This also allows us to sequentially train a model as new data is streamed in.
# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
# %% [markdown]
# ## Generate some dummy data
# We generate some ideal data for a linear regression model so that we know the true coefficients for this example.
# %%
n = 10000
m = 2
np.random.seed(3)

noise_std = 2
if 0:
    x = np.random.uniform(-1, 1, size=(n, m))
    w = np.random.uniform(-1, 1, size=(m, 1))
    b = np.random.normal(size=(1, 1))
    w = np.append(b, w, axis=0)
    y_true = w[0] + x @ w[1:]
    y = y_true + np.random.normal(loc=0, scale=noise_std, size=(n, 1))
else:
    x = np.random.uniform(-1, 1, size=(n, m))
    x = np.hstack([np.ones(shape=(n, 1)), x])
    w = np.random.uniform(-1, 1, size=(m + 1, 1))
    y_true = x @ w
    y = y_true + np.random.normal(loc=0, scale=noise_std, size=(n, 1))


x_train = x[: n // 2]
x_test = x[n // 2 :]
y_train = y[: n // 2]
y_test = y[n // 2 :]

plt.plot(w, ".")
plt.title("True coefficients")
plt.show()

# plt.plot(y_true, ".")
# plt.plot(y, ".")
# plt.title("True coefficients")
# plt.show()


# %%

results = sm.OLS(y, x).fit()
print(results.summary())
# extract coefficient distributions
w_sm_mu = results.params
w_sm_std = np.sqrt(np.diag(results.normalized_cov_params))

# %%
# time statsmodels with increasing data
import time

idx = np.floor(np.linspace(0, x.shape[0], num=50))[1:]
time_iter = []

params_mu = []
params_std = []
for end_idx in idx:
    t0 = time.process_time()
    results = sm.OLS(y[: int(end_idx)], x[: int(end_idx)]).fit()
    time_iter.append(time.process_time() - t0)

    params_mu.append(results.params)
    params_std.append(np.sqrt(np.diag(results.normalized_cov_params)))

# %%
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(idx, time_iter, label="statsmodels")
ax.set_ylabel("Time taken")
plt.legend()
plt.show()

# %%
params_mu = pd.DataFrame(params_mu, index=idx)
params_std = pd.DataFrame(params_std, index=idx, columns=params_mu.columns)


# %% [markdown]
# sequential bayesian regression
#
# $$\(p(\mathbf{w}\mid t) = \mathcal{N}(\mathbf{w}\mid \mathbf{m}_1,\mathbf{S}_1)\)$$
# with
# $$\(\mathbf{m}_1 = \mathbf{S}_1 (\mathbf{S}_0^{-1} \mathbf{m}_0 + \beta t \mathbf{x})\)$$
# $$\(\mathbf{S}_1 = \left(\mathbf{S}_0^{-1} + \beta\mathbf{x}\mathbf{x}^T\right)^{-1}\) $$
# %%
class BayesLinearRegressor:
    def __init__(
        self, number_of_features, mean=None, cov=None, alpha=1e6, beta=1
    ):
        # prior distribution on weights
        if mean is None:
            self.mean = np.array([[0] * (number_of_features)], dtype=np.float).T
            # self.mean = np.random.normal(loc=0.0, scale=1.0, size=(number_of_features,1))

        if cov is None:
            self.cov = alpha * np.identity(number_of_features)
            self.cov_inv = np.linalg.inv(self.cov)

        self.beta = beta  # process noise

    def fit(self, x, y):
        return self.update(x, y)

    def update(self, x, y):
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        if len(y.shape) == 1:
            y = y[:, np.newaxis]

        # https://cedar.buffalo.edu/~srihari/CSE574/Chap3/3.4-BayesianRegression.pdf
        cov_n_inv = self.cov_inv + self.beta * x.T @ x
        cov_n = np.linalg.inv(cov_n_inv)
        mean_n = cov_n @ (self.cov_inv @ self.mean + self.beta * x.T @ y)

        self.cov_inv = cov_n_inv
        self.cov = cov_n
        self.mean = mean_n

    def predict(self, x):
        mean = x @ self.mean
        scale = np.sqrt(np.sum(x @ self.cov @ x.T, axis=1))
        return mean, scale

    @property
    def coef_(self):
        return self.mean

    @property
    def scale_(self):
        return np.sqrt(np.diag(self.cov))


# %%
bayes_linear_regression = BayesLinearRegressor(x.shape[1])
bayes_linear_regression.fit(x, y)
np.testing.assert_array_almost_equal(
    bayes_linear_regression.coef_, params_mu.tail(1).transpose().to_numpy()
)
np.testing.assert_array_almost_equal(
    bayes_linear_regression.scale_, params_std.tail(1).to_numpy().flatten()
)

# %%
bayes_linear_regression = BayesLinearRegressor(x.shape[1])
for start_idx in range(x.shape[0]):
    print(start_idx)
    bayes_linear_regression.update(x[start_idx, :], y[[start_idx]])

np.testing.assert_array_almost_equal(
    bayes_linear_regression.coef_, params_mu.tail(1).transpose().to_numpy()
)
np.testing.assert_array_almost_equal(
    bayes_linear_regression.scale_, params_std.tail(1).to_numpy().flatten()
)

# %%
bayes_linear_regression = BayesLinearRegressor(x.shape[1], alpha=1)
prior_mu = bayes_linear_regression.coef_
prior_std = bayes_linear_regression.scale_


time_iter_seq = []
params_mu_seq = []
params_std_seq = []
for i, end_idx in enumerate(idx):
    t0 = time.process_time()
    if i > 0:
        start_idx = int(idx[i - 1])
    else:
        start_idx = 0
    bayes_linear_regression.update(
        x[start_idx : int(end_idx)],
        y[start_idx : int(end_idx)],
    )
    time_iter_seq.append(time.process_time() - t0)

    params_mu_seq.append(bayes_linear_regression.coef_.flatten())
    params_std_seq.append(bayes_linear_regression.scale_)

params_mu_seq = pd.DataFrame(
    params_mu_seq, index=idx, columns=params_mu.columns
)
params_std_seq = pd.DataFrame(
    params_std_seq, index=idx, columns=params_mu.columns
)

# %%
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(idx, time_iter, label="statsmodels")
ax.plot(idx, time_iter_seq, label="sequential")
ax.plot(idx, np.cumsum(time_iter_seq), label="cumulative_sequential")
ax.plot(idx, time_iter_fit, label="sequential_fit")
ax.set_ylabel("Time taken")
plt.legend()
plt.show()

# %%
params_mu_seq.plot()
params_std_seq.plot()
w
# %%
# plot posteriors vs prior
from scipy.stats import norm

x_range = np.linspace(-3, 3, num=1000)


def norm_max(x):
    return x / x.max()


fig, ax = plt.subplots(nrows=m + 1, figsize=(10, 6))
for idx in range(m + 1):
    ax[idx].plot(
        x_range,
        norm_max(norm.pdf(x_range, loc=prior_mu[idx], scale=prior_std[idx])),
    )
    ax[idx].plot(
        x_range,
        norm_max(
            norm.pdf(
                x_range,
                loc=params_mu_seq.iloc[-1, idx],
                scale=params_std_seq.iloc[-1, idx],
            )
        ),
    )

# %%
# plot results
bayes_linear_regression = BayesLinearRegressor(x.shape[1])
bayes_linear_regression.fit(x, y)
pred_mu, pred_scale = bayes_linear_regression.predict(x[:100])

sort_id = np.argsort(pred_mu.flatten())

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(pred_mu[sort_id].flatten(), ".")

ax.fill_between(
    np.arange(100),
    pred_mu[sort_id].flatten() - pred_scale[sort_id],
    pred_mu[sort_id].flatten() + pred_scale[sort_id],
    alpha=0.3,
)

# %%
