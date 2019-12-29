# %% [markdown]
# # Transport for London Cycle Data Exploration
#

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

# %%
# Regress on count data to find coefficient for weather conditions to get effect size, need to account for seasonality first
import statsmodels.api as sm
from patsy import dmatrices

y, X = dmatrices(
    "count ~ temp_feels + wind_speed + hum + is_weekend + is_raining",
    data=cycle_data,
    return_type="dataframe",
)

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())


# %%
# Seasonal trends
cycle_data.head()

# Adding cosine over the year - function of month
cycle_data["seasonal"] = 1 - np.cos((cycle_data["month"] - 1) / 12 * 2 * np.pi)

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
plt.scatter(cycle_data["month"], cycle_data["seasonal"], alpha=0.2)
# plt.xlabel("Date")
# plt.ylabel("Number of trips/day")
# plt.savefig("TFLCycles/images/against_time.png")
plt.show()

# %%
import statsmodels.api as sm
from patsy import dmatrices

y, X = dmatrices(
    "count ~ temp_feels + wind_speed + hum + is_weekend + is_raining + seasonal",
    data=cycle_data,
    return_type="dataframe",
)

model2 = sm.OLS(y, X)
results2 = model2.fit()
print(results2.summary())

# %% Check predictions
y_est = results2.predict(X)

# %% Check residuals
idx = range(0, 100)
fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
ax = fig.subplots()
ax.scatter(cycle_data["datetime"].loc[idx], y.loc[idx], alpha=0.2)
# ax.scatter(idx, y.loc[idx], alpha=0.2)
# plt.scatter(cycle_data["datetime"].loc[idx], y_est.loc[idx], alpha=0.2)
# fig.xlabel("Date")
# fig.ylabel("Number of trips/day")
# fig.legend(['Actual', 'Prediction'])


# format the ticks
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator())

# round to nearest years
# datemin = np.datetime64(cycle_day_data["datetime"].min(), "Y")
# datemax = np.datetime64(cycle_day_data["datetime"].max()) # + np.timedelta64(1, "Y")
# ax.set_xlim(datemin, datemax)

ax.grid(True)
fig.autofmt_xdate()


# fig = plt.figure()
# ax = fig.gca()
# ax.set_xticks(numpy.arange(0, 1, 0.1))
# ax.set_yticks(numpy.arange(0, 1., 0.1))
# plt.scatter(x, y)
# plt.grid()
# plt.show()

# plt.grid(b=None, axis='x')
# plt.savefig("TFLCycles/images/against_time.png")
plt.show()

# %%

# g = sns.FacetGrid(cycle_data, row="weather_code", col="is_weekend", margin_titles=True)
# bins = np.linspace(0, 60, 13)
# g.map(plt.hist, "cnt", color="steelblue", bins=bins)


# %%
# g = sns.pairplot(cycle_data)

