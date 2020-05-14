# %% [markdown]
# # Fitting one dimensional data
# Generate various one dimensional functions.
# Fit a neural network to them.
# Expand the domain of the x-axis, explore what happens.
# %%
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

import tensorflow as tf


# %%
# Exponential case
x = np.linspace(-5, 5, num=1000)
y = np.exp(x)

plt.plot(x, y)
plt.show()

# %%
# Log case
x = np.linspace(0, 5, num=1000)
x = x[2:]
y = np.log(x)

plt.plot(x, y)
plt.show()

# %%
# Model with one input for x, one output layer for y, hidden layer of 10 nodes
# Linear activation function

# Needs non linear activation if no polynomial inputs are given

# Taylor series model? Need polynomials of x

# Sigmoid activation allows for much smoother fits
# Relu is like miniture step funcitons

mode = 3
if mode == 0:
    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(4, input_shape=(1,)), tf.keras.layers.Dense(1)]
    )
if mode == 1:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                10, input_shape=(1,), activation=tf.keras.activations.relu
            ),
            tf.keras.layers.Dense(1),
        ]
    )
if mode == 2:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                10, input_shape=(1,), activation=tf.keras.activations.sigmoid
            ),
            tf.keras.layers.Dense(1),
        ]
    )
if mode == 3:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                10, input_shape=(1,), activation=tf.keras.activations.tanh
            ),
            tf.keras.layers.Dense(1),
        ]
    )

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1), loss=tf.keras.losses.mse
)
history = model.fit(x, y, epochs=50, verbose=False)

y_est = model.predict(x)

model.weights
# each layer = weights and bias
# each node output has a bias

plot_results(x, y, y_est, history.history["loss"])

# %% Plot results
def plot_results(x, y, y_est, loss):
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plt.plot(x, y, label="f(x)")
    plt.plot(x, y_est, label="NN")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    plt.subplot(122)
    plt.plot(loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()






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
