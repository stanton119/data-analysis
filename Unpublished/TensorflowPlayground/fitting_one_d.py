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

# Sigmoid activation allows for much smoother fits - dont use
# Tanh nearly always better than sigmoid (0 mean?)
# Relu is like miniture step funcitons - most common activation
#   Higher gradients than tanh means generally can converge faster
# LeakyRelu might be slightly better in practice, but is not commonly used

mode = 3
if mode == 0:
    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(4, input_shape=(1,)), tf.keras.layers.Dense(1)]
    )
if mode == 1:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                10, input_shape=(1,), activation=tf.keras.activations.relu,
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
                units=4,
                input_shape=(1,),
                activation=tf.keras.activations.tanh,
                kernel_regularizer=tf.keras.regularizers.l1(l=0.1),
            ),
            tf.keras.layers.Dense(
                units=1, kernel_regularizer=tf.keras.regularizers.l1(l=0.1)
            ),
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


plt.plot(model.weights[0].numpy().squeeze())
plt.plot(model.weights[2].numpy())
plt.show

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

