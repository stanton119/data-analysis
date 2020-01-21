# %% [markdown]
# # Transport for London Cycle Data Exploration
#
# Seasonality stuff
# https://otexts.com/fpp2/regression-intro.html
# https://stats.stackexchange.com/questions/204670/predict-seasonality-and-trend-combined-better-approach
# https://stats.stackexchange.com/questions/137995/is-there-a-way-to-allow-seasonality-in-regression-coefficients
# https://stats.stackexchange.com/questions/108877/capturing-seasonality-in-multiple-regression-for-daily-data

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.style.use("seaborn-whitegrid")

# %% Fetch data
dir_path = os.path.dirname(os.path.realpath(__file__))
os.path.join(dir_path, "data", "london_merged_processed.csv")

cycle_data = pd.read_csv(os.path.join(dir_path, "data", "london_merged_processed.csv"))

cycle_data.head()

# %% Correlation plots
sns.pairplot(cycle_data[["count", "temp_feels", "wind_speed", "hum"]])
# plt.savefig('TFLCycles/images/pairplot.png')
plt.show()

# %%
from holoviews.operation import gridmatrix
import holoviews as hv
from holoviews import opts

hv.extension("bokeh")
df = cycle_data[["count", "temp_feels", "wind_speed", "hum"]]

density_grid = gridmatrix(
    hv.Dataset(df), diagonal_type=hv.Distribution, chart_type=hv.Points
)
density_grid
# density_grid.opts(
#     opts.Scatter(
#         tools=["box_select", "lasso_select", "hover"],
#         border=0,
#         padding=0.1,
#         show_grid=True,
#     )
# )

# %% Extra features
cycle_data["is_raining"] = cycle_data["weather_code_label"] == "Rain"

# %% Initial regression model
# Regress on count data to find coefficient for weather conditions to get effect size, need to account for seasonality first
import statsmodels.api as sm
from patsy import dmatrices

y, X1 = dmatrices(
    "count ~ temp_feels + wind_speed + hum + is_weekend + is_raining",
    data=cycle_data,
    return_type="dataframe",
)

model = sm.OLS(y, X1)
results1 = model.fit()
print(results1.summary())


# %% Seasonal trends
cycle_data.head()

# Adding cosine over the year - function of month
cycle_data["seasonal"] = 1 - np.cos((cycle_data["month"] - 1) / 12 * 2 * np.pi)

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
plt.scatter(cycle_data["month"], cycle_data["seasonal"], alpha=0.2)
plt.xlabel("Date")
plt.ylabel("Seasonal feature")
# plt.savefig("TFLCycles/images/seasonal_feature.png")
plt.show()

# %% Refit model

y, X2 = dmatrices(
    "count ~ temp_feels + wind_speed + hum + is_weekend + is_raining + seasonal",
    data=cycle_data,
    return_type="dataframe",
)

model2 = sm.OLS(y, X2)
results2 = model2.fit()
print(results2.summary())

# %% Check predictions
y_est1 = results1.predict(X1)
y_est2 = results2.predict(X2)
cycle_data["y_est1"] = y_est1
cycle_data["y_est2"] = y_est2

# %% Check residuals
idx = range(0, y.shape[0])
fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
ax = fig.subplots()

if 0:
    ax.scatter(cycle_data["datetime"].loc[idx], y.loc[idx], alpha=0.2)
    ax.scatter(cycle_data["datetime"].loc[idx], y_est1.loc[idx], alpha=0.2)
    ax.scatter(cycle_data["datetime"].loc[idx], y_est2.loc[idx], alpha=0.2)
    plt.ylabel("Number of trips/day")
    plt.legend(["Actual", "Model1", "Model2"])
    fig_name = "pred"
else:
    ax.scatter(
        cycle_data["datetime"].loc[idx],
        y_est1.loc[idx] - y["count"].loc[idx],
        alpha=0.2,
    )
    ax.scatter(
        cycle_data["datetime"].loc[idx],
        y_est2.loc[idx] - y["count"].loc[idx],
        alpha=0.2,
    )
    plt.ylabel("Model residuals")
    plt.legend(["Model1", "Model2"])
    fig_name = "resid"
plt.xlabel("Date")


# format the ticks
import matplotlib.dates as mdates

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.grid(True)
fig.autofmt_xdate()

# plt.savefig(f"TFLCycles/images/{fig_name}.png")
plt.show()

# %% replacing plots with interactive versions
import hvplot.pandas

cycle_data["resid1"] = cycle_data["y_est1"] - cycle_data["count"]
cycle_data["resid2"] = cycle_data["y_est2"] - cycle_data["count"]
cycle_data.hvplot(
    y=["resid1", "resid2"], kind="scatter",
)

# %% Extreme residuals
filt_lim = cycle_data["resid2"].abs().nlargest(50).min()
filt = cycle_data["resid2"].abs() > filt_lim

# Seems some dates in particular are key
cycle_data.loc[filt][["datetime", "count", "y_est2", "resid2"]].head(n=filt.sum())
cycle_data.loc[filt][["datetime", "count", "y_est2", "resid2"]].head(n=50)
print(cycle_data.loc[filt]["datetime"])
# e.g. Christmas 2016 is really high?
# Christmas holidays are lower than usual

# New features to try:
# Is christmas period

# %% Residual histograms
fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
ax = fig.subplots()

ax.hist(y_est1.loc[idx] - y["count"].loc[idx], bins=30, alpha=0.5)
ax.hist(y_est2.loc[idx] - y["count"].loc[idx], bins=30, alpha=0.5)
plt.ylabel("Model residuals")
plt.legend(["Model1", "Model2"])
plt.xlabel("Date")
# plt.savefig("TFLCycles/images/resid_hist.png")
plt.show()

# %% Bootstrap fitting to check confidence intervals
y, X2 = dmatrices(
    "count ~ temp_feels + wind_speed + hum + is_weekend + is_raining + seasonal",
    data=cycle_data,
    return_type="dataframe",
)

n_bs = 1000
bs_len = y.shape[0]

model_coefs = []
for i in range(0, n_bs):
    y_temp = y.sample(n=bs_len, replace=True)
    X2_temp = X2.loc[y_temp.index]

    model2 = sm.OLS(y_temp, X2_temp)
    results2 = model2.fit()
    model_coefs.append(results2.params)

model_coefs = pd.DataFrame(model_coefs)

# estimate confidence range from bootstrap coefficients
print(model_coefs.quantile(q=[0.025, 0.975]))
print(model_coefs.mean())

# %% How do bootstrap coefficient converge with number of samples taken?
temp_hist = model_coefs["temp_feels"].hvplot(kind="hist")
temp_hist.opts(xlabel="temp_feels coefficient")


hv.save(temp_hist, "TFLCycles/images/temp_feels_hist.png")

# %%
import os

os.environ
os.environ["BOKEH_PHANTOMJS_PATH"] = "/Users/Rich/Developer/Data Science/VariousDataAnalysis/dataAnalysisEnv/lib/python3.7/site-packages/phantomjs_bin/bin/macosx"
        

# %%
print(results2.summary())

type(X2)
X2.shape

# %%
# g = sns.pairplot(cycle_data)

