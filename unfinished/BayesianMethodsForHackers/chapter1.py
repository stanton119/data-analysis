# %%
import numpy as np
import os
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")


import tensorflow as tf

import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


# %%
# Coin flip example
# Use bernoulli samples
# Build beta distributions from conjugate priors

# Build Graph
rv_coin_flip_prior = tfp.distributions.Bernoulli(probs=0.5, dtype=tf.int32)

num_trials = tf.constant([0, 1, 2, 3, 4, 5, 8, 15, 50, 500, 1000, 2000])

coin_flip_data = rv_coin_flip_prior.sample(num_trials[-1])

# prepend a 0 onto tally of heads and tails, for zeroth flip
coin_flip_data = tf.pad(coin_flip_data, tf.constant([[1, 0]]), "CONSTANT")

# compute cumulative headcounts from 0 to 2000 flips, and then grab them at each of num_trials intervals
cumulative_headcounts = tf.gather(tf.cumsum(coin_flip_data), num_trials)

# Creates a series of beta distributions at each num_trials
rv_observed_heads = tfp.distributions.Beta(
    concentration1=tf.cast(1 + cumulative_headcounts, tf.float32),
    concentration0=tf.cast(1 + num_trials - cumulative_headcounts, tf.float32),
)

probs_of_heads = tf.linspace(start=0.0, stop=1.0, num=100, name="linspace")
observed_probs_heads = tf.transpose(
    rv_observed_heads.prob(probs_of_heads[:, tf.newaxis])
)

# %%
# For the already prepared, I'm using Binomial's conj. prior.
plt.figure(figsize=(16, 9))
for i in range(len(num_trials)):
    sx = plt.subplot(len(num_trials) / 2, 2, i + 1)
    plt.xlabel("$p$, probability of heads") if i in [
        0,
        len(num_trials) - 1,
    ] else None
    plt.setp(sx.get_yticklabels(), visible=False)
    plt.plot(
        probs_of_heads,
        observed_probs_heads[i],
        label="observe %d tosses,\n %d heads"
        % (num_trials[i], cumulative_headcounts[i]),
    )
    plt.fill_between(probs_of_heads, 0, observed_probs_heads[i], alpha=0.4)
    plt.vlines(0.5, 0, 4, color="k", linestyles="--", lw=1)
    leg = plt.legend()
    leg.get_frame().set_alpha(0.4)
    plt.autoscale(tight=True)


plt.suptitle(
    "Bayesian updating of posterior probabilities", y=1.02, fontsize=14
)
plt.tight_layout()


# %%

x = tf.range(start=0.0, limit=16.0, dtype=tf.float32)
lambdas = tf.constant([1.5, 4.25])

poi_pmf = tfd.Poisson(rate=lambdas[:, tf.newaxis]).prob(x)

# %%
temp = tfd.Poisson(rate=2.5)
tf.reduce_mean(temp.sample(1000))

temp = tfd.Exponential(rate=2.5)
print(tf.reduce_mean(temp.sample(1000)))
print(1 / temp.rate)

# %%
a = tf.range(start=0.0, limit=4.0, delta=0.04)
a = a[..., tf.newaxis]
lambdas = tf.constant([0.5, 1.0])

# Now we use TFP to compute probabilities in a vectorized manner.
expo_pdf = tfd.Exponential(rate=lambdas).prob(a)


# %%

# Defining our Data and assumptions
# fmt: off
count_data = tf.constant([
    13,  24,   8,  24,   7,  35,  14,  11,  15,  11,  22,  22,  11,  57,  
    11,  19,  29,   6,  19,  12,  22,  12,  18,  72,  32,   9,   7,  13,  
    19,  23,  27,  20,   6,  17,  13,  10,  14,   6,  16,  15,   7,   2,  
    15,  15,  19,  70,  49,   7,  53,  22,  21,  31,  19,  11,  18,  20,  
    12,  35,  17,  23,  17,   4,   2,  31,  30,  13,  27,   0,  39,  37,   
    5,  14,  13,  22,
], dtype=tf.float32)
# fmt: on
n_count_data = tf.shape(count_data)
days = tf.range(n_count_data[0], dtype=tf.int32)

# %%
def joint_log_prob(count_data, lambda_1, lambda_2, tau):
    """
    Takes in data samples for count data and lambdas + tau
    Returns joint prob of all of those occurances, logged
    """
    # Estimate prior alpha as 1/average count
    alpha = 1.0 / tf.reduce_mean(count_data)
    rv_lambda_1 = tfd.Exponential(rate=alpha)
    rv_lambda_2 = tfd.Exponential(rate=alpha)

    rv_tau = tfd.Uniform()

    # Split days into different lambdas
    # Select lambda_1 for 0:tau-1, lambda_2 for tau:-1
    lambda_ = tf.gather(
        [lambda_1, lambda_2],
        indices=tf.cast(
            tau * tf.cast(tf.size(count_data), dtype=tf.float32)
            <= tf.cast(tf.range(tf.size(count_data)), dtype=tf.float32),
            dtype=tf.int32,
        ),
    )
    rv_observation = tfd.Poisson(rate=lambda_)

    return (
        rv_lambda_1.log_prob(lambda_1)
        + rv_lambda_2.log_prob(lambda_2)
        + rv_tau.log_prob(tau)
        + tf.reduce_sum(rv_observation.log_prob(count_data))
    )


# Define a closure over our joint_log_prob. - removes count_data as optimised parameter
def unnormalized_log_posterior(lambda1, lambda2, tau):
    return joint_log_prob(count_data, lambda1, lambda2, tau)


# %%
# wrap the mcmc sampling call in a @tf.function to speed it up
@tf.function(autograph=False)
def graph_sample_chain(*args, **kwargs):
    return tfp.mcmc.sample_chain(*args, **kwargs)


num_burnin_steps = 5000
num_results = 20000


# Set the chain's start state.
initial_chain_state = [
    tf.cast(tf.reduce_mean(count_data), tf.float32)
    * tf.ones([], dtype=tf.float32, name="init_lambda1"),
    tf.cast(tf.reduce_mean(count_data), tf.float32)
    * tf.ones([], dtype=tf.float32, name="init_lambda2"),
    0.5 * tf.ones([], dtype=tf.float32, name="init_tau"),
]


# Since HMC operates over unconstrained space, we need to transform the
# samples so they live in real-space.
unconstraining_bijectors = [
    tfp.bijectors.Exp(),  # Maps a positive real to R.
    tfp.bijectors.Exp(),  # Maps a positive real to R.
    tfp.bijectors.Sigmoid(),  # Maps [0,1] to R.
]

step_size = 0.2

kernel = tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_log_posterior,
        num_leapfrog_steps=2,
        step_size=step_size,
        state_gradients_are_stopped=True,
    ),
    bijector=unconstraining_bijectors,
)

kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    inner_kernel=kernel, num_adaptation_steps=int(num_burnin_steps * 0.8)
)


# Sample from the chain.
(
    [lambda_1_samples, lambda_2_samples, posterior_tau],
    kernel_results,
) = graph_sample_chain(
    num_results=num_results,
    num_burnin_steps=num_burnin_steps,
    current_state=initial_chain_state,
    kernel=kernel,
)

tau_samples = tf.floor(
    posterior_tau * tf.cast(tf.size(count_data), dtype=tf.float32)
)

plt.hist(lambda_1_samples, bins=100)
plt.hist(posterior_tau)

# %%
w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
plt.hist(
    tau_samples,
    bins=n_count_data[0],
    alpha=1,
    label=r"posterior of $\tau$",
    weights=w,
    rwidth=2.0,
)
plt.xticks(np.arange(n_count_data[0]))

plt.legend(loc="upper left")
plt.ylim([0, 0.75])
plt.xlim([35, len(count_data) - 20])
plt.xlabel(r"$\tau$ (in days)")
plt.ylabel(r"probability")

# %%
# Get samples of lambda1/2, tau for each day. Repeat 20k times. Use the posterior chains as samples
# Average lambda distribution per day and plot

# tau_samples, lambda_1_samples, lambda_2_samples contain
# N samples from the corresponding posterior distribution


N_ = tau_samples.shape[0]
expected_texts_per_day = tf.zeros(N_, n_count_data.shape[0])  # (10000,74)

plt.figure(figsize=(12.5, 9))

# Create matrix of days, copy over 20k samples
day_range = tf.range(0, n_count_data[0], delta=1, dtype=tf.int32)

# expand from shape of 74 to (10000,74)
day_range = tf.expand_dims(day_range, 0)
day_range = tf.tile(day_range, tf.constant([N_, 1]))

# Create multiple series of samples from tau, each day has 20k samples of tau
# expand from shape of 10000 to 10000,74
tau_samples_per_day = tf.expand_dims(tau_samples, 0)
tau_samples_per_day = tf.transpose(
    tf.tile(tau_samples_per_day, tf.constant([day_range.shape[1], 1]))
)

tau_samples_per_day = tf.cast(tau_samples_per_day, dtype=tf.int32)
# ix_day is (10000,74) tensor where axis=0 is number of samples, axis=1 is day. each value is true iff sampleXday value is < tau_sample value
ix_day = day_range < tau_samples_per_day

# Expand lambda samples over 20k samples, one sample pair per day
lambda_1_samples_per_day = tf.expand_dims(lambda_1_samples, 0)
lambda_1_samples_per_day = tf.transpose(
    tf.tile(lambda_1_samples_per_day, tf.constant([day_range.shape[1], 1]))
)
lambda_2_samples_per_day = tf.expand_dims(lambda_2_samples, 0)
lambda_2_samples_per_day = tf.transpose(
    tf.tile(lambda_2_samples_per_day, tf.constant([day_range.shape[1], 1]))
)

# Sum lambdas with binary on/off from tau
expected_texts_per_day = (
    tf.reduce_sum(
        lambda_1_samples_per_day * tf.cast(ix_day, dtype=tf.float32), axis=0
    )
    + tf.reduce_sum(
        lambda_2_samples_per_day * tf.cast(~ix_day, dtype=tf.float32), axis=0
    )
) / N_


# %% Add stats not just expectation
# My way
expected_texts_per_day_samples = tf.concat(
    [
        lambda_1_samples_per_day * tf.cast(ix_day, dtype=tf.float32),
        lambda_2_samples_per_day * tf.cast(~ix_day, dtype=tf.float32),
    ],
    axis=0,
)
# Removes 0s
expected_texts_per_day_samples
mask = tf.not_equal(expected_texts_per_day_samples, 0)
# different number of 0s per row
# need to transpose the mask and data and it iterates across rows
# reshape works along rows, so need long rows and then transpose back to full size
expected_texts_per_day_samples_mask = tf.transpose(
    tf.reshape(
        tf.boolean_mask(
            tf.transpose(expected_texts_per_day_samples), tf.transpose(mask)
        ),
        [-1, N_],
    )
)

# Find stats
expected_texts_per_day_mu = tf.reduce_mean(
    expected_texts_per_day_samples_mask, axis=0
)
expected_texts_per_day_std = tf.math.reduce_std(
    expected_texts_per_day_samples_mask, axis=0
)
expected_texts_per_day_std = tf.math.reduce
import numpy as np
expected_texts_per_day_quantiles = np.quantile(expected_texts_per_day_samples_mask.numpy(), [0.05, 0.95], axis=0)

assert (
    np.testing.assert_allclose(
        expected_texts_per_day.numpy(),
        expected_texts_per_day_mu.numpy(),
        rtol=1e-5,
    )
    is None
)

# %%
plt.figure(figsize=(12, 9))
plt.plot(
    range(n_count_data[0]),
    expected_texts_per_day_mu,
    lw=4,
    color="#E24A33",
    label="expected number of text-messages received",
)
if 0:
    plt.fill_between(
        range(n_count_data[0]),
        expected_texts_per_day_mu - expected_texts_per_day_std,
        expected_texts_per_day_mu + expected_texts_per_day_std,
        alpha=0.4,
    )
else:
    plt.fill_between(
        range(n_count_data[0]),
        expected_texts_per_day_quantiles[0],
        expected_texts_per_day_quantiles[1],
        alpha=0.4,
        color="#0000DA",
        label='ConfidenceInterval [0.05, 0.95]'
    )
    
plt.xlim(0, n_count_data.numpy()[0])
plt.xlabel("Day")
plt.ylabel("Expected # text-messages")
plt.title("Expected number of text-messages received")
plt.ylim(0, 60)
plt.bar(
    np.arange(len(count_data)),
    count_data,
    color="#5DA5DA",
    alpha=0.65,
    label="observed texts per day",
)

plt.legend(loc="upper left")


# %%
