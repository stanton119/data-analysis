# %%
import numpy as np
import pandas as pd
from time import process_time, time
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf

plt.style.use("seaborn-whitegrid")

import utilities


# %% Generate data
df = utilities.gen_ar_data()
df = utilities.get_stock_data()

df = utilities.get_weather_data()

df_train, df_test = utilities.split_ts(df)
df_train.plot()
df_test.plot()

# split into training samples
x_train, y_train = utilities.split_sequence(
    df=df_train, y_col="y", train_len=60, forecast_gap=170, forecast_len=24
)
x_test, y_test = utilities.split_sequence(
    df=df_test, y_col="y", train_len=60, forecast_gap=170, forecast_len=24
)

# tf approach
forecast_gap = 170
train_len = 60
forecast_len = 24
window_df = tf.keras.preprocessing.timeseries_dataset_from_array(
    df_train["y"].iloc[:-forecast_gap].to_numpy(),
    df_train["y"].iloc[forecast_gap:].to_numpy(),
    sequence_length=train_len,
)


for batch in window_df:
    inputs, targets = batch


# x_train.shape = (n, train_len, n_features), y_train.shape = (n, forecast_len, n_outputs)
_, train_len, n_features = x_train.shape
n_outputs = y_train.shape[1]

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
# 
# 
# 
# 
# 
# 
# 
# 

# %% Predict
y_train_hat = model.predict(x_train, verbose=1)
y_test_hat = model.predict(x_test, verbose=1)

# %% Plot results
def construct_results_df(df, y, y_hat):
    df_results = pd.DataFrame(index=df.index)
    df_results["y"] = df["y"]
    df_results["tf_y"] = np.nan
    df_results["tf_y"].iloc[train_len : -n_outputs - 1] = y[:, 0].flatten()

    for col in range(y_hat.shape[1]):
        df_results[f"y_hat_{col}"] = np.nan
        df_results[f"y_hat_{col}"].iloc[
            train_len + col : -n_outputs - 1 + col
        ] = y_hat[:, col]

        # df_results[f'mse_{col}'] = (df_results[f'y_hat_{col}'] - df_results['y'])**2
    return df_results


y_train.shape
y_train[:, 0].shape
df_train.iloc[train_len : -n_outputs - 1].shape
df_results_train = construct_results_df(df_train, y_train, y_train_hat)
df_results_test = construct_results_df(df_test, y_test, y_test_hat)

df_results_train.plot()
df_results_test.plot()

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
