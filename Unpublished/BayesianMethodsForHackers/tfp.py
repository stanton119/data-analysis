
# %%
import tensorflow as tf
import tensorflow_probability as tfp



# %%
# Pretend to load synthetic data set.
features = tfp.distributions.Normal(loc=0.0, scale=1.0).sample(int(100e3))
labels = tfp.distributions.Bernoulli(logits=1.618 * features).sample()

# Specify model.
model = tfp.glm.Bernoulli()

# Fit model given data.
coeffs, linear_response, is_converged, num_iter = tfp.glm.fit(
    model_matrix=features[:, tf.newaxis],
    response=tf.cast(labels, dtype=tf.float32),
    model=model,
)
# ==> coeffs is approximately [1.618] (We're golden!)


# %%
import matplotlib.pyplot as plt

plt.hist(features)
plt.hist(labels)

tf.math.reduce_mean(features)
tf.math.reduce_mean(labels)
tf.math.mean(labels)

tf.math.exp(1.618)
tf.math.log(1.618)


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

# %% sklearn data
import sklearn.datasets

n = 1000
m = 10
noise = 0.1
x, y, w = sklearn.datasets.make_regression(
    n_samples=n,
    n_features=m,
    n_informative=int(np.round(m / 2)),
    coef=True,
    noise=noise,
)

# %% TF tutorial data
# @title Synthesize dataset.
w0 = 0.125
b0 = 5.0
x_range = [-20, 60]


def load_dataset(n=150, n_tst=150):
    np.random.seed(43)

    def s(x):
        g = (x - x_range[0]) / (x_range[1] - x_range[0])
        return 3 * (0.25 + g ** 2.0)

    x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
    eps = np.random.randn(n) * s(x)
    y = (w0 * x * (1.0 + np.sin(x)) + b0) + eps
    x = x[..., np.newaxis]
    x_tst = np.linspace(*x_range, num=n_tst).astype(np.float32)
    x_tst = x_tst[..., np.newaxis]
    return y, x, x_tst


y, x, x_tst = load_dataset()

# %% Define support dists
# Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.0))
    return tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.Independent(
                    tfp.distributions.Normal(
                        loc=t[..., :n],
                        scale=1e-5 + tf.nn.softplus(c + t[..., n:]),
                    ),
                    reinterpreted_batch_ndims=1,
                )
            ),
        ]
    )


# Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.Independent(
                    tfp.distributions.Normal(loc=t, scale=1),
                    reinterpreted_batch_ndims=1,
                )
            ),
        ]
    )


# %%
model_params = {}

# %%
# Static model
model = tf.keras.Sequential([tf.keras.layers.Dense(1), tf.keras.layers.Dense(1), tf.keras.layers.Dense(1)])
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=tf.keras.losses.mse
)
model.fit(x, y, epochs=500, verbose=False)

model_params['static'] = model.weights

# %%
# Build model.


negloglik = lambda y, p_y: -p_y.log_prob(y)


for i in [1, 2, 3, 4]:

    if i==1:
        # static model with normal distribution mean output
        model_name = 'static_normal'
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(1),
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.Normal(loc=t, scale=1)
                ),
            ]
        )
    if i==2:
        # Two outputs at final layer, model the output uncertainty. Fixed coefficients
        model_name = 'static_normal_std'
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(1 + 1),
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.Normal(
                        loc=t[..., :1],
                        scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]),
                    )
                ),
            ]
        )
    if i==3:
        # One output at final layer, bayesian coefficients.
        model_name = 'bayesian_normal'
        model = tf.keras.Sequential(
            [
                tfp.layers.DenseVariational(
                    1, posterior_mean_field, prior_trainable
                ),
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.Normal(loc=t, scale=1)
                ),
            ]
        )
    if i==4:
        model_name = 'bayesian_normal_std'
        model = tf.keras.Sequential([
            tfp.layers.DenseVariational(1 + 1, posterior_mean_field, prior_trainable, kl_weight=1/x.shape[0]),
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.Normal(loc=t[..., :1],
                                    scale=1e-3 + tf.math.softplus(0.01 * t[...,1:]))),
            ])

    print(i, model_name)
    # Do inference.
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
    model.fit(x, y, epochs=500, verbose=False)

    model_params[model_name] = model.weights

model.weights
model.trainable_weights

# %%
w
model_params

# %% Make predictions.
import matplotlib.pyplot as plt

# model(x) is not deterministic for the DenseVaritational case
yhat = model(x)
yhat.mean()
yhat.stddev()

plt.plot(x, y, ".")
plt.plot(x, yhat.mean(), ".")
plt.plot(x, yhat.mean() + 2.0 * yhat.stddev(), ".")
plt.plot(x, yhat.mean() - 2.0 * yhat.stddev(), ".")
plt.grid()
plt.show()

# %%
# model(x) is not deterministic for the DenseVaritational case


plt.plot(x, y, ".")
[plt.plot(x, model(x).mean(), ".") for i in range(100)]
plt.grid()
plt.show()

