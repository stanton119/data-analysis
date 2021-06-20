# %%
"""
Fit a VAE to time series data
Generate novel time series from the latent space
"""


# %%
import numpy as np
import pandas as pd
from time import process_time, time

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

# %% Load stocks data
df = pd.read_csv("aapl_us.csv")
df.dtypes
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")
df["Open"].plot()

# %%
def gen_time_series(df: pd.Series, ts_len: int = 500) -> np.array:
    start_idx = np.random.randint(low=0, high=len(df) - ts_len)
    return df.iloc[start_idx : start_idx + ts_len].to_numpy()


# %% Generate loads of time series
x_ts = []
n = 100
for ii in range(n):
    x_ts.append(gen_time_series(df["Open"]))
x_ts = np.stack(x_ts).transpose()

# %% Scale
ts = x_ts[:,np.random.randint(n)]
def scale_ts(ts:np.array)->np.array:
    return (ts - ts.mean())/ts.std()

plt.plot(ts)
plt.plot(scale_ts(ts))

for ii in range(n):
    x_ts[:,ii] = scale_ts(x_ts[:,ii])
x_ts.mean(axis=0)
x_ts.std(axis=0)

# %% Train/test split
test_frac = 0.2
x_ts_train = x_ts[:, :-int(n*0.2)]
x_ts_test = x_ts[:, -int(n*0.2):]

# %%
"""
%load_ext autoreload
%autoreload 2
"""


import vae as models

vae, encoder, decoder = models.create_vae_model_2()


_ = vae.fit(train_dataset,
            epochs=15,
            validation_data=eval_dataset)


# %% Create dummy data
if 1:
    df = pd.DataFrame()
    df["ds"] = pd.date_range(start="2010-01-01", end="2025-01-01", freq="1D")
    df["y"] = np.random.rand(df.shape[0], 1)
    df["x1"] = np.random.rand(df.shape[0], 1)
else:
    data_location = (
        "https://raw.githubusercontent.com/ourownstory/neural_prophet/master/"
    )
    df = pd.read_csv(data_location + "example_data/wp_log_peyton_manning.csv")

df_train = df.iloc[: int(df.shape[0] / 2)]
df_test = df.iloc[int(df.shape[0] / 2) :]

# %% AR data
def gen_time_series():
    df = pd.DataFrame()
    df["ds"] = pd.date_range(start="2010-01-01", end="2025-01-01", freq="1D")
    freq = np.random.rand() * 20
    df["x1"] = np.sin(
        np.linspace(start=0, stop=freq * 2 * np.math.pi, num=df.shape[0])
    )
    freq = np.random.rand() * 20
    df["x2"] = np.sin(
        np.linspace(start=0, stop=freq * 2 * np.math.pi, num=df.shape[0])
    )
    df["y"] = df["x1"] + df["x2"]
    return df


# %% Generate loads of time series
x_ts = []
for ii in range(100):
    df = gen_time_series()
    x_ts.append(df["y"].to_numpy())
x_ts = np.stack(x_ts).transpose()

# plt.plot(x_ts[:,3])

# %%

# %% Fit VAE


# %% Explore latent space


# %% Fit CNN
df.set_index("ds")["y"].plot()

df_train = df.iloc[: int(df.shape[0] / 2)]
df_test = df.iloc[int(df.shape[0] / 2) :]
# %%
# univariate data preparation
from numpy import array

# split a univariate sequence into samples
def split_sequence(sequence, n_steps, n_forecast=1):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1 - n_forecast:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix + n_forecast - 1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# define input sequence
# choose a number of time steps
n_steps = 3
n_features = 1
n_forecast = 50
# split into samples
X_train, y_train = split_sequence(df_train["y"].to_numpy(), n_steps, n_forecast)


def shape_tf_input(X):
    return X.reshape((X.shape[0], X.shape[1], n_features))


# %%
import tensorflow as tf

# define model
model = tf.keras.Sequential()
model.add(
    tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=2,
        activation="relu",
        input_shape=(n_steps, n_features),
    )
)
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(20, activation="relu"))
model.add(tf.keras.layers.Dense(1))

model.compile(
    loss="mse",
    optimizer=tf.optimizers.Adam(),
    # optimizer=tf.optimizers.SGD(learning_rate=0.01),
    # metrics=["mae"],
)
# model.compile(optimizer='adam', loss='mse')

model.summary()

# %%
# early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=20, monitor="val_loss", restore_best_weights=True
)

# fit model
history = model.fit(
    shape_tf_input(X_train),
    y_train,
    validation_split=0.25,
    batch_size=64,
    epochs=1000,
    verbose=True,
    callbacks=[early_stopping],
)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

# %% Predict
y_train_hat = model.predict(shape_tf_input(X_train), verbose=1)

df_results_train = pd.DataFrame(index=df_train["ds"])
df_results_train["y"] = df_train["y"].to_numpy()
df_results_train["y_hat"] = np.nan
df_results_train.loc[:, "y_hat"].iloc[
    n_steps + n_forecast :
] = y_train_hat.flatten()
df_results_train

df_results_train.plot()

X_test, y_test = split_sequence(df_test["y"].to_numpy(), n_steps, n_forecast)
y_test_hat = model.predict(shape_tf_input(X_test), verbose=1)

df_results_test = pd.DataFrame(index=df_test["ds"])
df_results_test["y"] = df_test["y"].to_numpy()
df_results_test["y_hat"] = np.nan
df_results_test.loc[:, "y_hat"].iloc[
    n_steps + n_forecast :
] = y_test_hat.flatten()
df_results_test

df_results_test.plot()

# %%
