# %% [markdown]
# # Fitting a Distribution with Pyro
#
# Generate random data from a gaussian with some mean and standard deviation.
# We want to estimate a distribution that best fits the data using variational inference with Pyro.
#
# References:
#   * http://pyro.ai/examples/intro_part_ii.html
#
# Import the required libraries:
# %%
import numpy as np
import torch
from torch.distributions import constraints
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

from scipy.stats import norm
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

# %% [markdown]
# ## Generate observed data
# We use `numpy` to generate random data from a Gaussian distribution with a known mean and standard deviation.
# %% Generate observed data
n = 1000
x = np.random.randn(n, 1)
std = 4
mu = 2
x = x * std + mu

print(f"Shape: {x.shape}")
print(f"Mean: {np.mean(x)}")
print(f"Standard deviation: {np.std(x)}")


# %% [markdown]
# ```
# Shape: (1000, 1)
# Mean: 1.987280841100236
# Standard deviation: 3.9196571939982863
# ```

# %% [markdown]
# Now that we have generated the data we need to use it to estimate the original distribution parameters.
#
# The traditional approach, when using a Gaussian distribution, would be to simply take the mean and standard deviation, as we did above. The following approach, however, will give us a full distribution rather than a point estimate, and it should generalise to many different problems.
#
# ## Variational inference
# This is solving a bayesian posterior estimation problem with gradient descent.
#
# We specify the type of distribution the posterior should follow. We then tune the parameters to minimise the difference between the estimated posterior and the actual posterior. The difference measure used is the KL divergence. (KL divergence is not symmetrical, not sure why its the way round it is)
#
# $$D_{\mathrm{KL}}(Q \parallel P) \triangleq \sum_\mathbf{Z}  Q(\mathbf{Z}) \log \frac{Q(\mathbf{Z})}{P(\mathbf{Z}\mid \mathbf{X})}$$
#
# It starts with a class of approximating distributions. Then it finds the best approximation to the posterior distribution. It minimise the Kullback-Leibler divergence between our approximate distribution and the posterior. This is equivalent to maximising the evidence lower bound (ELBO). This requires calculating the joint distribution, rather than the true posterior.
#
# This is now an optimisation problem which can be solved with gradient descent algorithms.
#
# This is different to MCMC. With MCMC we get a numerical approximation to the exact posterior using a set of samples, Variational Bayes provides a locally-optimal, exact analytical solution to an approximation of the posterior.


# References:
# * https://www.youtube.com/watch?v=3KGZDC3-_iY
# * http://pyro.ai/examples/bayesian_regression.html
# * https://en.wikipedia.org/wiki/Variational_Bayesian_methods

# %% [markdown]
# ## Pyro approach
# `pyro` is a probabilistic library that sits on top of `pytorch` that enables variational inference.
# To solve this problem in pyro we need a few different components.
#
# We need a model of our data generating function.

# %% [markdown]
# ### Data generating function
# We assume a Guassian distribution as the model to generate our random data.
# This function takes parameters for our distributions and generates a random sample from the resulting distribution.
# Our model consists of a Gaussian distribution which has two priors: mean and standard deviation.
# These parameters come from distributions themselves.
# The mean is taken from another Gaussian distribution. The standard deviation comes from a Gamma distribution.
#
# This is represented as:
# $$x\sim\mathcal{N}\left(\mu,\sigma^{2}\right)$$
# $$\mu\sim\mathcal{N}\left(\mu_{\mu},\mu_{\sigma^{2}}\right)$$
# $$\sigma\sim\mathrm{Gamma}\left(\alpha,\beta\right)$$
#

# The parameters for these two distributions ($\mu_{\mu},\mu_{\sigma}, \alpha,\beta$) are the function inputs:
# ```
#     params: [mu_prior + std_prior]
#         mu_prior - Gaussian - mu, std
#         std_prior - Gamma - a, b
# ```

# We then condition the function, so that the samples produced from `data_dist` are enforced to match those from our original random data, `x`.

# %%
def data_model(params):
    mu_dist = pyro.sample("mu_dist", dist.Normal(params[0], params[1]))
    std_dist = pyro.sample(
        "std_dist", dist.Gamma(np.abs(params[2]), np.abs(params[3]))
    )
    return pyro.sample("data_dist", dist.Normal(mu_dist, std_dist))


conditioned_data_model = pyro.condition(
    data_model, data={"data_dist": torch.tensor(x.flatten())}
)


# %% [markdown]
# ### Guide function
# The guide function represents the family of distribution we want to consider as our posterior distribution, therefore it should be an approximation of the model posterior distribution. In this case we assume a Guassian distribution as the approximating class for the posterior distribution. This is an ideal case as we know the original data came from a Gaussian, in practice this would be based on domain knowledge.
#
# The guide has two requirements:
# * The guide function must take the same parameters as the generating model.
# * The data seen from the model must be valid outputs from the guide function.
#
#
# These functions are built with pyro primatives so that they can be used with gradient descent to optimise the KL divergence.
# The function params are in the same form as the above data generating model.
# The `pyro.param` statements recall the named parameters from the pyro param store. If no parameter exists with that name it will use the `param[.]` value passed to it, this happens on the first call only.

# We use the constraint property to ensure the distribution parameters are correctly $>0$.
#
# We use the `torch.abs` calls to ensure the distribution parameters are correctly $>0$.
#
# We make both `mu_dist` and `std_dist` as separate objects in order to optimise the mean and standard deviation of our data separately.

# %%
def parametrised_guide(params):
    mu_mu = pyro.param("mu_mu", torch.tensor(params[0]))
    mu_std = pyro.param(
        "mu_std", torch.tensor(params[1]), constraint=constraints.positive
    )
    std_a = pyro.param(
        "std_a", torch.tensor(params[2]), constraint=constraints.positive
    )
    std_b = pyro.param(
        "std_b", torch.tensor(params[3]), constraint=constraints.positive
    )

    mu_dist = pyro.sample("mu_dist", dist.Normal(mu_mu, mu_std))
    std_dist = pyro.sample("std_dist", dist.Gamma(std_a, std_b))
    return pyro.sample("data_dist", dist.Normal(mu_dist, std_dist))


# %% [markdown]
# ### Setup variational inference descent
# This is setup via the object `pyro.infer.SVI()` using the functions we generated above.
#
# We use stochastic gradient descent. This is parameterised by the learning rate and momentum. These values were picked by trial and error so that it converges well.
#
# The loss function to optimise is the evidence lower bound.

# %%
svi = pyro.infer.SVI(
    model=conditioned_data_model,
    guide=parametrised_guide,
    optim=pyro.optim.SGD({"lr": 0.00001, "momentum": 0.8}),
    loss=pyro.infer.Trace_ELBO(),
)

# %% [markdown]
# ### Prior initialisation
# We choose uninformed priors for the mean (Gaussian) and standard deviation (Gamma) prior distributions.
# This suggests we want to learn from the data without assuming any significant previous knowledge.
#
# ### Gradient descent
# Starting with our priors we iterate over our data. Each iteration we step the gradient descent optimiser.
# This should push our estimated posterior distribution closer to the actual posterior from the data each time.
# At each step we store the parameters so we can inspect them afterwards.

# %%
mu_prior = [0.0, 10.0]  # Gaussian - mu, std
std_prior = [1.0, 0.1]  # Gamma - a, b
params_prior = mu_prior + std_prior

# Iterate over all the data
losses, mu_mu, mu_std, std_a, std_b = [], [], [], [], []
pyro.clear_param_store()

# %%
num_steps = 5000
for t in range(num_steps):
    losses.append(svi.step(params_prior))
    mu_mu.append(pyro.param("mu_mu").item())
    mu_std.append(pyro.param("mu_std").item())
    std_a.append(pyro.param("std_a").item())
    std_b.append(pyro.param("std_b").item())


# %% [markdown]
# ### Results
# The loss function has reduced with time:

# %%
# Convergence of the loss function
plt.plot(losses)
plt.title("ELBO")
plt.xlabel("Iteration")
plt.ylabel("loss")
plt.savefig("images/elbo.png")
plt.show()

# %% [markdown]
# ![](images/elbo.png)
#
# We can also see how the distribution parameters have converged:

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
plt.savefig("images/params.png")
plt.show()

# %% [markdown]
# ![](images/params.png)
#
# The parameters for the mean distribution have converged well. The parameters of the standard deviation distribution have behaved differently.
#
# First, we can show the PDF of the mean distribution comparing the prior and posteriors:


# %%
# Plot mean distributions
mu_prior_dist = norm(loc=mu_prior[0], scale=mu_prior[1])
x_range = np.linspace(mu_prior_dist.ppf(0.01), mu_prior_dist.ppf(0.99), num=100)
y_values = mu_prior_dist.pdf(x_range)
plt.plot(x_range, y_values, label="prior")

mu_post_dist = norm(loc=mu_mu[-1], scale=mu_std[-1])
x_range = np.linspace(mu_post_dist.ppf(0.01), mu_post_dist.ppf(0.99), num=100)
y_values = mu_post_dist.pdf(x_range)
plt.plot(x_range, y_values, label="posterior")

plt.xlabel("x")
plt.ylabel("prob(x)")
plt.title("Mean PDF")
plt.legend()
plt.savefig("images/mean_dist.png")
plt.show()

# %% [markdown]
# ![](images/mean_dist.png)
#
# The prior is mostly flat, the posterior on the other hand is very sharp.
# It is very confident that the actual mean is around 2, which would be correct.
# The standard deviation of our mean distribution, `mu_std[-1]=0.20`, is on a similar scale to what we would expect from the sample error of the mean: `np.std(x) / np.sqrt(n) = 0.12`.

#
# Similarly we look at the distribution of the standard deviation.
# %%
# Plot std distributions
x_range = np.linspace(0, 10, num=100)

std_prior_dist = dist.Gamma(std_prior[0], std_prior[1])
y_values = torch.exp(std_prior_dist.log_prob(x_range))
plt.plot(x_range, y_values, label="prior")

std_post_dist = dist.Gamma(std_a[-1], std_b[-1])
y_values = torch.exp(std_post_dist.log_prob(x_range))
plt.plot(x_range, y_values, label="posterior")

plt.title("Standard Deviation PDF")
plt.legend()
plt.savefig("images/std_dist.png")
plt.show()

# %% [markdown]
# ![](images/std_dist.png)
#
# The prior is similarly mostly flat. The posterior has a peak around 3.8 which is close to the true value of 4, or the sample standard deviation of 3.9.
#
# To look into the non-converging parameters let's look at distribution at different points in its training:

# %% Previous std dists.
x_range = np.linspace(0, 10, num=100)

for idx in [500, 1000, 2000, 3000, 4000, 4999]:
    std_post_dist = dist.Gamma(std_a[idx], std_b[idx])
    y_values = torch.exp(std_post_dist.log_prob(x_range))
    plt.plot(x_range, y_values, label=idx)

plt.title("Standard Deviation PDF")
plt.legend()
plt.savefig("images/std_dist_idx.png")
plt.show()

# %% [markdown]
# ![](images/std_dist_idx.png)
#
# The distribution is converging towards the correct value from about 1000 iterations. The parameters kept changing in the same direction. As the distributions are converging this suggests that the two parameters $\alpha, \beta$ are some what correlated, allowing both to change to improve our loss function. This can cause the optimisation to struggle or take longer. I will not pursue this much further though, as the distribution has converged well.
#
# The data distribution can be plotted over the original data to see a goodness of fit:

# %% Plot estimated distribution over original data
plt.hist(x, density=True)

# plot prior
prior_mu = mu_prior[0]
prior_std = std_prior[0] / std_prior[1]  # distribution mean
prior_dist = norm(loc=prior_mu, scale=prior_std)
x_range = np.linspace(prior_dist.ppf(0.01), prior_dist.ppf(0.99), num=100)
y_values = prior_dist.pdf(x_range)
plt.plot(x_range, y_values, label="prior")

# plot posterior
post_mu = mu_mu[-1]
post_std = std_a[-1] / std_b[-1]
post_dist = norm(loc=post_mu, scale=post_std)
x_range = np.linspace(post_dist.ppf(0.01), post_dist.ppf(0.99), num=100)
y_values = post_dist.pdf(x_range)
plt.plot(x_range, y_values, label="post")

plt.legend()
plt.title("Data histogram")
plt.savefig("images/data_dist.png")
plt.show()

print(post_mu)
print(post_std)

# %% [markdown]
# ![](images/data_dist.png)
#
# The posterior (green line) fits the data histogram well as we would expect.

# The values of the posterior distribution (1.98, 3.79) are similar to those from the sample estimates (1.99, 3.92).
# However in the posterior case we have our confidence around those values rather than just point estimates.
