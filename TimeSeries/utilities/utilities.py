from typing import List, Tuple
import pandas as pd
import numpy as np
from tqdm import trange

# processing functions
def split_ts(
    df: pd.DataFrame, split_frac: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df.iloc[: int(df.shape[0] * split_frac)]
    df_test = df.iloc[int(df.shape[0] * split_frac) :]
    return df_train, df_test


def split_sequence(
    df, y_col: str, train_len=30, forecast_gap=1, forecast_len=3
):
    # split time series into training samples
    # forecast_gap = distance to first forecast

    x, y = list(), list()
    len_df = df.shape[0]
    for i in trange(len_df):
        end_ix = i + train_len
        if end_ix > len_df - 1 - forecast_gap - forecast_len:
            break
        seq_x = df.iloc[i:end_ix, :]  # .drop(columns=y_col)
        seq_y = df.iloc[
            end_ix + forecast_gap - 1 : end_ix + forecast_gap - 1 + forecast_len
        ][[y_col]]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


def construct_results_df(df, y, y_hat, train_len, forecast_gap, forecast_len):
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
        ] = y_hat[:, col].flatten()

        # df_results[f'mse_{col}'] = (df_results[f'y_hat_{col}'] - df_results['y'])**2
    return df_results


def one_step_ar_predict(
    model, train_len, no_forecasts: int, initial_x: np.array, scaler=None
):
    y_test_hat_ar = initial_x
    for i in trange(no_forecasts):

        # transform if possible
        if scaler is not None:
            _x_test = scaler.transform(y_test_hat_ar[-train_len:, np.newaxis])
        else:
            _x_test = y_test_hat_ar[-train_len:, np.newaxis]

        _y_test_hat = model.predict(_x_test.reshape(1, train_len, 1), verbose=1)
        y_test_hat_ar = np.append(y_test_hat_ar, _y_test_hat)

    return y_test_hat_ar
