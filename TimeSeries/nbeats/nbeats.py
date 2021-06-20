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
forecast_gap = 100
train_len = 60
forecast_len = 24

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
x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(
    x_test.shape
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
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

# %% Predict
y_train_hat = model.predict(x_train, verbose=1)
y_test_hat = model.predict(x_test, verbose=1)

# %% one step pred
# keep first test sample, predict next, append to array, repeat
y_test_hat = model.predict(x_test, verbose=1)


# %% Plot results
df_results_train = utilities.construct_results_df(
    df_train, y_train, y_train_hat, train_len, forecast_gap, forecast_len
)
df_results_train.plot()


df_results_test = utilities.construct_results_df(
    df_test, y_test, y_test_hat, train_len, forecast_gap, forecast_len
)

fig, ax = plt.subplots(figsize=(10,6))
df_results_train.iloc[:,:5].plot(ax=ax)
df_results_train.iloc[:,-2:].plot(ax=ax)

df_results_test.iloc[:,:5].plot()
df_results_test.iloc[:,-5:].plot()






# %%

from pytorch_forecasting import (
    TimeSeriesDataSet,
    NBeats,
    TemporalFusionTransformer,
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_forecasting.data import NaNLabelEncoder

# %% nbeats model
# create dataset and dataloaders
max_encoder_length = 60
max_prediction_length = 20

df_train_nbeats = df_train.copy()
df_train_nbeats = df_train_nbeats.reset_index()
df_train_nbeats = df_train_nbeats.reset_index()
df_train_nbeats["group"] = 0

df_train_nbeats_sub, df_train_nbeats_val = utilities.split_ts(df_train_nbeats)

nbeats_training = TimeSeriesDataSet(
    df_train_nbeats_sub,
    time_idx="index",
    target="y",
    categorical_encoders={
        "group": NaNLabelEncoder().fit(df_train_nbeats_sub["group"])
    },
    group_ids=["group"],
    time_varying_unknown_reals=["y"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
)
nbeats_validation = TimeSeriesDataSet.from_dataset(
    nbeats_training, df_train_nbeats_val
)

# %%
batch_size = 128
nbeats_train_dataloader = nbeats_training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0
)
nbeats_val_dataloader = nbeats_validation.to_dataloader(
    train=False, batch_size=batch_size, num_workers=0
)

net = NBeats.from_dataset(
    nbeats_training,
    learning_rate=3e-2,
    weight_decay=1e-2,
    widths=[32, 512],
    backcast_loss_ratio=0.1,
)

early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
)
trainer = pl.Trainer(
    max_epochs=100,
    gpus=0,
    weights_summary="top",
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback],
    limit_train_batches=30,
)

trainer.fit(
    net,
    train_dataloader=nbeats_train_dataloader,
    val_dataloaders=nbeats_val_dataloader,
)

# %%
import torch
from pytorch_forecasting import Baseline

actuals = torch.cat([y[0] for x, y in iter(nbeats_val_dataloader)])

next(iter(nbeats_val_dataloader))
baseline_predictions = Baseline().predict(nbeats_val_dataloader)
SMAPE()(baseline_predictions, actuals)


# %%
trainer = pl.Trainer(gpus=0, gradient_clip_val=0.1)
net = NBeats.from_dataset(
    training,
    learning_rate=3e-2,
    weight_decay=1e-2,
    widths=[32, 512],
    backcast_loss_ratio=0.1,
)


early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min"
)
lr_logger = LearningRateMonitor()
trainer = pl.Trainer(
    max_epochs=100,
    gpus=0,
    gradient_clip_val=0.1,
    limit_train_batches=30,
    callbacks=[lr_logger, early_stop_callback],
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=32,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=7,
    loss=QuantileLoss(),
    log_interval=2,
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# find optimal learning rate
res = trainer.lr_find(
    tft,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
    early_stop_threshold=1000.0,
    max_lr=0.3,
)

print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()

trainer.fit(
    tft, train_dataloader=train_dataloader, val_dataloaders=val_dataloader,
)


# %%

forecast_gap = 170
train_len = 60
forecast_len = 24

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

# tf approach
if 0:
    window_df = tf.keras.preprocessing.timeseries_dataset_from_array(
        df_train["y"].iloc[:-forecast_gap].to_numpy(),
        df_train["y"].iloc[forecast_gap:].to_numpy(),
        sequence_length=train_len,
    )

    for batch in window_df:
        inputs, targets = batch


# x_train.shape = (n, train_len, n_features), y_train.shape = (n, forecast_len, forecast_len)
# _, train_len, n_features = x_train.shape
# forecast_len = y_train.shape[1]

# scale inputs
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(
    x_train.shape
)
x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(
    x_test.shape
)


# %%

# define model
model = tf.keras.Sequential()
model.add(
    tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=30,
        activation="relu",
        input_shape=(train_len, n_features),
    )
)
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(50, activation="relu"))
model.add(tf.keras.layers.Dense(forecast_len))

model.compile(
    loss="mse",
    optimizer=tf.optimizers.Adam(),
    # optimizer=tf.optimizers.SGD(learning_rate=0.01),
    # metrics=["mae"],
)

model.summary()

x_train.shape
x_test.shape

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

# %% Predict
y_train_hat = model.predict(x_train, verbose=1)
y_test_hat = model.predict(x_test, verbose=1)


# %% Plot results
def construct_results_df(df, y, y_hat):
    df_results = pd.DataFrame(index=df.index)
    df_results["y"] = df["y"]
    df_results["tf_y"] = np.nan
    df_results["tf_y"].iloc[
        train_len + forecast_gap - 1 : -forecast_len - 1
    ] = y[:, 0].flatten()

    for col in range(y_hat.shape[1]):
        df_results[f"y_hat_{col}"] = np.nan
        df_results[f"y_hat_{col}"].iloc[
            train_len + forecast_gap - 1 + col : -forecast_len - 1 + col
        ] = y_hat[:, col]

        # df_results[f'mse_{col}'] = (df_results[f'y_hat_{col}'] - df_results['y'])**2
    return df_results


y_train.shape
y_train[:, 0].shape
df_train.iloc[train_len : -forecast_len - 1].shape
df_results_train = construct_results_df(df_train, y_train, y_train_hat)
df_results_test = construct_results_df(df_test, y_test, y_test_hat)

df_results_train.iloc[:, :5].plot()
df_results_test.iloc[:, :5].plot()

# %% Metrics
def construct_metrics_df(df_results):
    df_metrics = pd.DataFrame(index=df.index)

    for col in df_results.columns:
        if "hat" in col:
            df_metrics[f"mse_{col}"] = (df_results[col] - df_results["y"]) ** 2
    return df_metrics


df_metrics_train = construct_metrics_df(df_results_train)
df_metrics_test = construct_metrics_df(df_results_test)

mse_cols = [col for col in df_results_train.columns if "mse" in col]
hat_cols = [col for col in df_results_train.columns if "hat" in col]

df_metrics_train.cumsum().plot()
df_metrics_test.cumsum().plot()

df_results_train.plot()
df_results_test.plot()
