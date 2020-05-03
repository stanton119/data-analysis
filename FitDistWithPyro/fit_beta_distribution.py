# %% [markdown]
# # Fitting a Distribution with Pyro
#
# In this simple example we will fit a Gaussian distribution to random data from a gaussian with some known mean and standard deviation.
# We want to estimate a distribution that best fits the data using variational inference with Pyro.
#
# References:
#   * [https://pyro.ai/examples/intro_part_ii.html](https://pyro.ai/examples/intro_part_ii.html)
#
# Import the required libraries:
# %% Fit beta posterior
import numpy as np
import torch
from torch.distributions import constraints
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
torch.manual_seed(0)
# %% [markdown]
# ## Generate observed data
# We use the distribution module of pytorch to generate random data from Bernoulli trials with a known probability of success $$P(p)=0.4$$.
# %%
# Generate data from actual distribution
true_dist = dist.Bernoulli(0.4)
n = 100
data = true_dist.sample(sample_shape=(n, 1))
# %% [markdown]
# ## Generate observed data
# We use the distribution module of pytorch to generate random data from Bernoulli trials with a known probability of success $$P(p)=0.4$$.
# %%
# Prior distribution
prior = dist.Beta(10, 10)

# %%
# Analytical posterior
posterior = dist.Beta(
    prior.concentration1 + data.sum(),
    prior.concentration0 + len(data) - data.sum(),
)



# %%
if 0:
    posterior = prior
    idx = 0
    block_idx = np.arange(len(data)).reshape(10, -1)

# %%
if 0:
    # Step one at a time
    # print(data[:idx])
    data_temp = data[block_idx[idx]]
    posterior = dist.Beta(
        posterior.concentration1 + data_temp.sum(),
        posterior.concentration0 + len(data_temp) - data_temp.sum(),
    )
    idx += 1

    plt.figure(num=None, figsize=(10, 6), dpi=80)
    x_range = np.linspace(0, 1, num=100)
    y_values = torch.exp(posterior.log_prob(torch.tensor(x_range)))
    plt.plot(x_range, y_values, label="posterior")
    y_values = torch.exp(prior.log_prob(torch.tensor(x_range)))
    plt.plot(x_range, y_values, label="prior")
    plt.title("PDF")
    plt.legend()
    # plt.savefig("images/std_dist.png")
    plt.show()

# %%
# Variational inference
def data_model(params):
    # returns a Bernoulli trial outcome
    beta = pyro.sample("beta_dist", dist.Beta(params[0], params[1]))
    return pyro.sample("data_dist", dist.Bernoulli(beta))


conditioned_data_model = pyro.condition(data_model, data={"data_dist": data})


def guide(params):
    # returns the Bernoulli probablility
    alpha = pyro.param(
        "alpha", torch.tensor(params[0]), constraint=constraints.positive
    )
    beta = pyro.param(
        "beta", torch.tensor(params[1]), constraint=constraints.positive
    )
    return pyro.sample("beta_dist", dist.Beta(alpha, beta))


svi = pyro.infer.SVI(
    model=conditioned_data_model,
    guide=guide,
    optim=pyro.optim.SGD({"lr": 0.001, "momentum": 0.8}),
    loss=pyro.infer.Trace_ELBO(),
)

params_prior = [prior.concentration1, prior.concentration0]

# Iterate over all the data
losses, alpha, beta = [], [], []
pyro.clear_param_store()

num_steps = 3000
for t in range(num_steps):
    losses.append(svi.step(params_prior))
    alpha.append(pyro.param("alpha").item())
    beta.append(pyro.param("beta").item())

posterior_vi = dist.Beta(alpha[-1], beta[-1])

# %%
# plt.plot(alpha)
# plt.plot(beta)
# plt.plot(losses)


# %%
# Plot results
plt.figure(num=None, figsize=(10, 6), dpi=80)
x_range = np.linspace(0, 1, num=100)

y_values = torch.exp(prior.log_prob(torch.tensor(x_range)))
plt.plot(x_range, y_values, label="prior")

y_values = torch.exp(posterior.log_prob(torch.tensor(x_range)))
plt.plot(x_range, y_values, label="posterior")

y_values = torch.exp(posterior_vi.log_prob(torch.tensor(x_range)))
plt.plot(x_range, y_values, label="posterior_vi")

plt.title("PDF")
plt.legend()
plt.savefig("images/beta_pdfs.png")
plt.show()
