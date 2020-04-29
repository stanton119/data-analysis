# %% [markdown]
# # Pyro
#
# ## Fitting a distribution
# Generate random data from a gaussian with some mean and standard deviation
#
# Mirrors: http://pyro.ai/examples/intro_part_ii.html

# %% Generate observed data
import numpy as np

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
import torch
import torch.optim as optim
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist


# %% Create model
# Initial guesses - priors
# We assume a Guassian distribution as the approximating class for the posterior distribution
# The hyperparameters (mu and std) are specified by their conjugate priors, Gaussian and Gamma.
# The distribution for mu suggests we are uninformed.
# Initially we set the standard deviations to a fix constant

mu_prior = [0.0, 10.0]  # Gaussian - mu, std
std_prior = [1.0, 10.0]  # Gamma


def data_model(guess):

    mu_dist = pyro.sample("mu_dist", dist.Normal(guess[0], guess[1]))
    return pyro.sample("data_dist", dist.Normal(mu_dist, std_prior[0]))


# This step forces the samples of the above model to output $x$
conditioned_data_model = pyro.condition(data_model, data={"data_dist": x[0][0]})
conditioned_data_model = pyro.condition(data_model, data={"data_dist": torch.tensor(x.flatten())})

# This is the same as setting the obs value of the data_dist object:
# data_dist = pyro.sample(
#     "data_dist", pyro.distributions.Normal(mu_dist, std_prior[0]), obs=x
# )

conditioned_data_model(mu_prior)


# %% Infering latent distributions
# Building a guide function
# The guide function should be an approximation of the model posterior distribution.
# The guide function must take the same parameters as the generating model
# The data seen from the model must be valid outputs from the guide function
# For the above we chose the Gaussian distribution to approximate the posterior.

# We specify a family of guides and chose the best one for the posterior fit.


def parametrized_guide(guess):
    a = pyro.param("a", torch.tensor(guess[0]))
    b = pyro.param("b", torch.tensor(guess[1]))
    return pyro.sample("mu_dist", dist.Normal(a, torch.abs(b)))  # to force a positive std


parametrized_guide(mu_prior)

# %%
mu_init = 0.0
mu_init = mu_prior

pyro.clear_param_store()
svi = pyro.infer.SVI(
    model=conditioned_data_model,
    guide=parametrized_guide,
    optim=pyro.optim.SGD({"lr": 0.00001, "momentum": 0.1}),
    loss=pyro.infer.Trace_ELBO(),
)

# %%
losses, a, b = [], [], []
num_steps = 2500
for t in range(num_steps):
    losses.append(svi.step(mu_init))
    a.append(pyro.param("a").item())
    b.append(pyro.param("b").item())

# %%
import matplotlib.pyplot as plt

plt.plot(losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss")
print("a = ", pyro.param("a").item())
print("b = ", pyro.param("b").item())


# %%
plt.subplot(1, 2, 1)
plt.plot(a)
plt.ylabel("a")
plt.grid()

plt.subplot(1, 2, 2)
plt.ylabel("b")
plt.plot(b)
plt.grid()
plt.tight_layout()


# %%
# Plot estimated distribution over original data

# %% Train with mini batch gradient descent