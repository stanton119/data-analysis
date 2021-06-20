"""
Predict next sequence element
When predicting feed result back to get next prediction. Repeat
"""

# %%
import numpy as np
import pandas as pd
from time import process_time, time
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import utilities
import data

# %load_ext autoreload
# %autoreload 2

plt.style.use("seaborn-whitegrid")


# %% Generate data
df = data.get_weather_data()

df_train, df_test = utilities.split_ts(df)
df_train.plot()
df_test.plot()


# %% keras nbeats
forecast_gap = 1
train_len = 600
forecast_len = 1

# should do split_ts after forming training samples to avoid removing data

# split into training samples
x_train, y_train = utilities.split_sequence(
    df=df_train,
    y_col="y",
    train_len=train_len,
    forecast_gap=forecast_gap,
    forecast_len=forecast_len,
)
x_test, y_test = utilities.split_sequence(
    df=df_test,
    y_col="y",
    train_len=train_len,
    forecast_gap=forecast_gap,
    forecast_len=forecast_len,
)

# x_train.shape = (n, train_len, n_features), y_train.shape = (n, forecast_len, forecast_len)
# _, train_len, n_features = x_train.shape
# forecast_len = y_train.shape[1]

# scale inputs
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(
    x_train.shape
)

# %% Fit model

import nbeats_model
import tensorflow as tf

model = nbeats_model.NBeatsNet(
    backcast_length=train_len,
    forecast_length=forecast_len,
    stack_types=(
        nbeats_model.NBeatsNet.GENERIC_BLOCK,
        nbeats_model.NBeatsNet.GENERIC_BLOCK,
    ),
    nb_blocks_per_stack=2,
    thetas_dim=(4, 4),
    share_weights_in_stack=True,
    hidden_layer_units=64,
)

model.compile(
    loss="mse", optimizer=tf.optimizers.Adam(learning_rate=1e-5),
)

model.summary()

# %%
# early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=20, monitor="val_loss", restore_best_weights=True
)

# fit model
history = model.fit(
    x_train,
    y_train,
    validation_split=0.25,
    batch_size=64,
    epochs=1000,
    verbose=True,
    callbacks=[early_stopping],
)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

# %% Predict single step
y_train_hat = model.predict(x_train, verbose=1)

x_test_whole = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(
    x_test.shape
)
y_test_hat = model.predict(x_test_whole, verbose=1)

# %% Predict multiple steps
# keep first test sample, predict next, append to array, repeat
# y_test_hat = model.predict(x_test, verbose=1)

y_test_hat_ar = utilities.one_step_ar_predict(
    model,
    train_len,
    no_forecasts=y_test.shape[0],
    initial_x=x_test[0].flatten(),
    scaler=scaler,
)

# %% Predict multiple steps
# ideally need to refactor the model to retain its internal state with LSTMs etc.
# predictions tend toward 0 seasonality
y_test_hat_ar_long = utilities.one_step_ar_predict(
    model=model,
    train_len=train_len,
    no_forecasts=2000,
    initial_x=df_train['y'].to_numpy(),
    # initial_x=x_test[0].flatten(),
    scaler=scaler,
)

plt.plot(y_test_hat_ar_long)


# %%
df_results_train = utilities.construct_results_df(
    df_train, y_train, y_train_hat, train_len, forecast_gap, forecast_len
)
df_results_train.plot()

df_results_test = utilities.construct_results_df(
    df_test, y_test, y_test_hat, train_len, forecast_gap, forecast_len
)
df_results_test_ar = utilities.construct_results_df(
    df_test,
    y_test,
    y_test_hat_ar[-y_test.shape[0] :, np.newaxis, np.newaxis],
    train_len,
    forecast_gap,
    forecast_len,
)

df_results_test["y_hat_ar"] = df_results_test_ar["y_hat_0"]
df_results_test.plot()
