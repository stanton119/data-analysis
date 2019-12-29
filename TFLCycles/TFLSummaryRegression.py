# %% [markdown]
# # Transport for London Cycle Data Exploration
#
# Seasonality stuff
https://otexts.com/fpp2/regression-intro.html
https://stats.stackexchange.com/questions/204670/predict-seasonality-and-trend-combined-better-approach
https://stats.stackexchange.com/questions/137995/is-there-a-way-to-allow-seasonality-in-regression-coefficients
https://stats.stackexchange.com/questions/108877/capturing-seasonality-in-multiple-regression-for-daily-data

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.style.use("seaborn-whitegrid")


# %%
# Fetch data
cycle_data = pd.read_csv("TFLCycles/data/london_merged_processed.csv")

cycle_data.head()

# %% Correlation plots
sns.pairplot(cycle_data[["count", "temp_feels", "wind_speed", "hum"]])
# plt.savefig('TFLCycles/images/pairplot.png')
plt.show()

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

# %% Residual histograms
fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
ax = fig.subplots()

ax.hist(y_est1.loc[idx] - y["count"].loc[idx], bins=30, alpha = 0.5)
ax.hist(y_est2.loc[idx] - y["count"].loc[idx], bins=30, alpha = 0.5)
plt.ylabel("Model residuals")
plt.legend(["Model1", "Model2"])
plt.xlabel("Date")
# plt.savefig("TFLCycles/images/resid_hist.png")
plt.show()

# %%

# g = sns.FacetGrid(cycle_data, row="weather_code", col="is_weekend", margin_titles=True)
# bins = np.linspace(0, 60, 13)
# g.map(plt.hist, "cnt", color="steelblue", bins=bins)


# %%
# g = sns.pairplot(cycle_data)

