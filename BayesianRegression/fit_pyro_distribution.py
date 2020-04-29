# %% [markdown]
# # Pyro
#
# ## Fitting a distribution
# Generate random data from a gaussian with some mean and standard deviation
#
# Mirrors: http://pyro.ai/examples/intro_part_ii.html

# %%
import numpy as np

import torch
import torch.optim as optim
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

from scipy.stats import norm
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

# %% Generate observed data

n = 1000
x = np.random.randn(n, 1)
std = 4
mu = 2
x = x * std + mu


print(x.shape)
print(np.mean(x))
print(np.std(x))


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
# * Given an estimate of the model weights, we predict what our output, $y$, should be.
# * Calculate a loss function based on the difference between our prediction and the actual output. We use the above MSE function.
# * Update the model weights to improve the loss function.
# * Iterate the above until the weights have converged.
#
# We will apply this in pytorch.
# Pytorch has its own internal memory structures so we need to convert from our numpy arrays to torch tensors using `from_numpy()`. The weights estimates are initialised randomly. We require that the gradients are calculated for the weights so we use the `requires_grad` flag.
#
# We use the stochastic gradient descent optimiser to update the weights, the learning rate needs to be chosen appropriately.
#
# The forward step calculates the output of the network. The loss function is setup using pytorch functions which run over the tensor objects and allow the gradients to be calculated automatically.
#
# The gradients need to be reset each iteration as PyTorch accumulates the gradients on subsequent backward passes. The backwards step calculates the gradients automatically.
#
# The optimizer object then updates the model weights to minimise the loss function.
#
# We iterate over the data 100 times, at which point the weights have converged.

# %%


# %% Create model
# Initial guesses - priors
# We assume a Guassian distribution as the approximating class for the posterior distribution
# The hyperparameters (mu and std) are specified by their conjugate priors, Gaussian and Gamma.
# The distribution for mu suggests we are uninformed.
# Initially we set the standard deviations to a fix constant

mu_prior = [0.0, 10.0]  # Gaussian - mu, std
std_prior = [1.0, 0.1]  # Gamma - a, b
params_prior = mu_prior + std_prior


def data_model(params):
    mu_dist = pyro.sample("mu_dist", dist.Normal(params[0], params[1]))
    std_dist = pyro.sample("std_dist", dist.Gamma(np.abs(params[2]), np.abs(params[3])))
    return pyro.sample("data_dist", dist.Normal(mu_dist, std_dist))


# This step forces the samples of the above model to output $x$
conditioned_data_model = pyro.condition(
    data_model, data={"data_dist": torch.tensor(x.flatten())}
)

# This is the same as setting the obs value of the data_dist object:
# data_dist = pyro.sample(
#     "data_dist", pyro.distributions.Normal(mu_dist, std_prior[0]), obs=x
# )

conditioned_data_model(params_prior)


# %% Infering latent distributions
# Building a guide function
# The guide function should be an approximation of the model posterior distribution.
# The guide function must take the same parameters as the generating model
# The data seen from the model must be valid outputs from the guide function
# For the above we chose the Gaussian distribution to approximate the posterior.

# We specify a family of guides and chose the best one for the posterior fit.


def parametrized_guide(params):
    mu_mu = pyro.param("mu_mu", torch.tensor(params[0]))
    mu_std = pyro.param("mu_std", torch.tensor(params[1]))
    std_a = pyro.param("std_a", torch.tensor(params[2]))
    std_b = pyro.param("std_b", torch.tensor(params[3]))
    if 0:
        return pyro.sample(
            "mu_dist", dist.Normal(mu_mu, torch.abs(mu_std))
        )  # to force a positive std
    else:
        mu_dist = pyro.sample("mu_dist", dist.Normal(mu_mu, torch.abs(mu_std)))
        std_dist = pyro.sample(
            "std_dist", dist.Gamma(torch.abs(std_a), torch.abs(std_b))
        )
        return pyro.sample("data_dist", dist.Normal(mu_dist, torch.abs(std_dist)))


parametrized_guide(params_prior)

# %%
pyro.clear_param_store()
svi = pyro.infer.SVI(
    model=conditioned_data_model,
    guide=parametrized_guide,
    optim=pyro.optim.SGD({"lr": 0.0001, "momentum": 0.5}),
    loss=pyro.infer.Trace_ELBO(),
)

# %%
# Iterate over all the data
losses, mu_mu, mu_std, std_a, std_b = [], [], [], [], []
num_steps = 2500
for t in range(num_steps):
    losses.append(svi.step(params_prior))
    mu_mu.append(pyro.param("mu_mu").item())
    mu_std.append(pyro.param("mu_std").item())
    std_a.append(pyro.param("std_a").item())
    std_b.append(pyro.param("std_b").item())

# %%

plt.plot(losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss")
print("mu_mu = ", pyro.param("mu_mu").item())
print("mu_std = ", pyro.param("mu_std").item())
print("std_a = ", pyro.param("std_a").item())
print("std_b = ", pyro.param("std_b").item())


# %% Parameters against iteration
plt.subplot(2, 2, 1)
plt.plot(mu_mu)
plt.ylabel("mu_mu")

plt.subplot(2, 2, 2)
plt.ylabel("mu_std")
plt.plot(mu_std)

plt.subplot(2, 2, 3)
plt.ylabel("std_a")
plt.plot(std_a)

plt.subplot(2, 2, 4)
plt.ylabel("std_b")
plt.plot(std_b)



# %%
# Plot mean distributions
# Very confident in the tuned mu distribution
mu_prior_dist = norm(loc=mu_prior[0], scale=mu_prior[1])
x_range = np.linspace(mu_prior_dist.ppf(0.01), mu_prior_dist.ppf(0.99), num=100)
y_values = mu_prior_dist.pdf(x_range)
plt.plot(x_range, y_values, label='prior')

mu_post_dist = norm(loc=pyro.param("mu_mu").item(), scale=pyro.param("mu_std").item())
x_range = np.linspace(mu_post_dist.ppf(0.01), mu_post_dist.ppf(0.99), num=100)
y_values = mu_post_dist.pdf(x_range)
plt.plot(x_range, y_values, label='posterior')

plt.xlabel('x')
plt.ylabel('prob(x)')
plt.title('Mean PDF')
plt.legend()

# %%
# Plot std distributions
# Converges on 4
x_range = np.linspace(0, 10, num=100)

std_prior = [1, 0.1]

std_prior_dist = dist.Gamma(std_prior[0], std_prior[1])
y_values = torch.exp(std_prior_dist.log_prob(x_range))
plt.plot(x_range, y_values, label='prior')

std_post_dist = dist.Gamma(pyro.param("std_a").item(), pyro.param("std_b").item())
y_values = torch.exp(std_post_dist.log_prob(x_range))
plt.plot(x_range, y_values, label='posterior')

plt.title('Standard Deviation PDF')
plt.legend()


# %%
# Plot estimated distribution over original data


plt.hist(x, density=True)


prior_mu = mu_prior[0]
prior_std = (std_prior[0]-1)/std_prior[1]    # distribution mode
prior_std = (std_prior[0])/std_prior[1]      # distribution mean
prior_dist = norm(loc=prior_mu, scale=prior_std)
x_range = np.linspace(prior_dist.ppf(0.01), prior_dist.ppf(0.99), num=100)
y_values = prior_dist.pdf(x_range)
plt.plot(x_range, y_values, label='prior')

post_mu = pyro.param("mu_mu").item()
post_std = (pyro.param("std_a").item()-1)/pyro.param("std_b").item()    # distribution mode
post_std = (pyro.param("std_a").item())/pyro.param("std_b").item()      # distribution mean
post_dist = norm(loc=post_mu, scale=post_std)
x_range = np.linspace(post_dist.ppf(0.01), post_dist.ppf(0.99), num=100)
y_values = post_dist.pdf(x_range)
plt.plot(x_range, y_values, label='post')

plt.legend()
plt.title('Data')

# %% Train with mini batch gradient descent
