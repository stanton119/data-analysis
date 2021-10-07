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

