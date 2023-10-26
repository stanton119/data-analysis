# %% [markdown]
# # Neural prophet - ARNet
# Build a model to predict multiple future time steps.
# This is broken at the moment. Increasing `n_forecasts` > 1 causes the library to crash.
#
# Import some relevant libraries:
# %%
import numpy as np
import pandas as pd
from time import process_time, time
import matplotlib.pyplot as plt

from fbprophet import Prophet
from neuralprophet import NeuralProphet

plt.style.use("seaborn-whitegrid")
# %%
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "utilities"))
import utilities
import data

# %%
#############################################################
# %% AR data
df = pd.DataFrame()
df["ds"] = pd.date_range(start="2010-01-01", end="2025-01-01", freq="1D")
freq = 10
df["x1"] = np.sin(np.linspace(start=0, stop=freq * 2 * np.math.pi, num=df.shape[0]))
freq = 3
df["x2"] = np.sin(np.linspace(start=0, stop=freq * 2 * np.math.pi, num=df.shape[0]))
df["y"] = df["x1"] + df["x2"]

df.set_index("ds")["y"].plot()

df_train = df.iloc[: int(df.shape[0] / 2)]
df_test = df.iloc[int(df.shape[0] / 2) :]

# %%
t1 = process_time()
model_nprophet = NeuralProphet()
model_nprophet = NeuralProphet(n_lags=100, n_forecasts=10)
model_nprophet.add_future_regressor("x1")
model_nprophet.add_future_regressor("x2")
model_nprophet.fit(df_train, freq="D")
t2 = process_time() - t1

t3 = process_time()
future_nprophet = model_nprophet.make_future_dataframe(
    df=df_train,  # .iloc[[-1]],
    regressors_df=df_test[["x1", "x2"]],
    periods=df_test.shape[0],
)
df_pred_nprophet = model_nprophet.predict(future_nprophet)
t4 = process_time() - t3
print(t2, t4)

# df_pred_nprophet.set_index('ds')['yhat1'].plot()

# fig1 = model_nprophet.plot(df_pred_nprophet)

# %%
t1 = process_time()
model_nprophet = NeuralProphet(n_lags=100, n_forecasts=10)
model_nprophet.fit(df_train[["ds", "y"]], freq="D")
t2 = process_time() - t1

t3 = process_time()
future_nprophet = model_nprophet.make_future_dataframe(df=df_train[["ds", "y"]])
df_pred_nprophet = model_nprophet.predict(future_nprophet)

t4 = process_time() - t3
print(t2, t4)

# df_pred_nprophet.set_index('ds')['yhat1'].plot()

fig1 = model_nprophet.plot(df_pred_nprophet)

# %%
