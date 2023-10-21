# %% [markdown]
# # Fitting ConvNN to Datasets
# MNIST/EMNIST
# Explore layers with NN microscope
# Add data augmentation
# %%
import tensorflow as tf
import tensorflow_datasets as tfds
import os

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

# %% Load data from csv
import pandas as pd
df_train = pd.read_csv('data/mnist/train.csv')
df_test = pd.read_csv('data/mnist/test.csv')

def split_data(df: pd.DataFrame):
    return df.drop(columns=['label']).to_numpy(), df['label'].to_numpy()

def preprocess_train_data(data: np.array):
    return (data/255.).astype('float32')


# %% Build model
"""
With cnn we should be able to get similar accuracy with fewer weights
"""


mode = 0
if mode == 0:
    model = tf.keras.models.Sequential(
        [
            # tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
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
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=["accuracy"],
)

# %%
x_train, y_train = split_data(df_train)
history = model.fit(
    x=x_train, y=y_train, epochs=6, validation_split=0.3, batch_size=64, verbose=True
)

y_est = model.predict(ds_train)

# %% Train in cv
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
idx_train, idx_test = next(kf.split(df_train))
for idx_train, idx_test in kf.split(df_train):
    x_train, y_train = split_data(df_train.iloc[idx_train])
    x_test, y_test = split_data(df_train.iloc[idx_test])
    x_train = preprocess_train_data(x_train)
    x_test = preprocess_train_data(x_test)

    history = model.fit(
        x=x_train, y=y_train, epochs=6, validation_data=(x_test, y_test), batch_size=64, verbose=True
    )

    y_train_est = model.predict(x_train)
    y_est = model.predict(x_test)



# %%
# i=0

i += 1
plt.imshow(np.reshape(x_test[i], newshape=(28,28)).squeeze())

plt.title(f"{y_test[i]}, {np.argmax(y_est[i])}")
plt.show()

# %%
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Train')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.show()

# %%
preprocess_train_data
y_est.shape
y_est[0]
np.argmax(y_est[0])+1
np.mean(y_est, axis=0)

# %% Show confusion matrix
cov = np.cov(y_est, rowvar=False)
cov = np.abs(np.corrcoef(cov, rowvar=False))
if 1:
    indx, indy = np.diag_indices(len(cov))
    cov[indx, indy] = 0

fig, ax = plt.subplots(figsize=(8,12))
im = ax.imshow(cov)
ax.set_xticks(np.arange(len(cov)))
ax.set_yticks(np.arange(len(cov)))
# ... and label them with the respective list entries
ax.set_xticklabels(ds_info.features['label'].names)
ax.set_yticklabels(ds_info.features['label'].names)

# Loop over data dimensions and create text annotations.
for i in range(len(cov)):
    for j in range(len(cov)):
        text = ax.text(j, i, np.round(cov[i, j], decimals=3),
                       ha="center", va="center", color="w")

ax.set_title("confusion matrix")
fig.tight_layout()
plt.show()




# %%
import matplotlib.pyplot as plt
plt.plot(history.history["loss"])

# %%
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
