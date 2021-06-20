"""
LSTM

LSTM - encoder/decoder
https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
"""

# %%
import numpy as np
import pandas as pd
from time import process_time, time
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import utilities
import data

plt.style.use("seaborn-whitegrid")


# %% Generate data
df = data.get_weather_data()

df_train, df_test = utilities.split_ts(df)
df_train.plot()
df_test.plot()


# %% keras nbeats
forecast_gap = 1
train_len = 60
forecast_len = 14

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
_, _, n_features = x_train.shape
# forecast_len = y_train.shape[1]

# scale inputs
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(
    x_train.shape
)
x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(
    x_test.shape
)

# %% Fit model
import tensorflow as tf

if 1:
    # single LSTM
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.LSTM(
            200, activation="relu", input_shape=(train_len, n_features)
        )
    )
    model.add(tf.keras.layers.Dense(100, activation="relu"))
    model.add(tf.keras.layers.Dense(forecast_len))
else:
    # encoder/decoder
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.LSTM(
            200, activation="relu", input_shape=(train_len, n_features)
        )
    )
    model.add(tf.keras.layers.RepeatVector(forecast_len))
    # decoder - returns whole sequence, not just final value
    model.add(
        tf.keras.layers.LSTM(200, activation="relu", return_sequences=True)
    )
    model.add(
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(100, activation="relu")
        )
    )
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))

model.compile(
    loss="mse", optimizer=tf.optimizers.Adam(),
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
plt.plot(history.history["loss"][1:])
plt.plot(history.history["val_loss"][1:])

# %% Predict
y_train_hat = model.predict(x_train, verbose=1)
y_test_hat = model.predict(x_test, verbose=1)

# %% Plot results
df_results_train = utilities.construct_results_df(
    df_train, y_train, y_train_hat, train_len, forecast_gap, forecast_len
)
df_results_train.plot()

df_results_test = utilities.construct_results_df(
    df_test, y_test, y_test_hat, train_len, forecast_gap, forecast_len
)

fig, ax = plt.subplots(figsize=(10, 6))
df_results_train.iloc[:, :3].plot(ax=ax)
df_results_train.iloc[:, -1:].plot(ax=ax)

fig, ax = plt.subplots(figsize=(10, 6))
df_results_test.iloc[:, :3].plot(ax=ax)
df_results_test.iloc[:, -1:].plot(ax=ax)

fig, ax = plt.subplots(figsize=(10, 6))
df_results_test.iloc[-100:, :3].plot(ax=ax)
df_results_test.iloc[-100:, -1:].plot(ax=ax)


# %% Metrics
((df_results_test['y_hat_0'] - df_results_test['y'])**2).mean()
((df_results_test.iloc[:,-1] - df_results_test['y'])**2).mean()
