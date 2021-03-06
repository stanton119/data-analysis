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
cycle_data = pd.read_csv("TFLCycles/data/london_merged.csv")
# data retrieved from: https://www.kaggle.com/hmavrodiev/london-bike-sharing-dataset

# Change column names
col_mappings = {"cnt": "count", "t1": "temperature", "t2": "temp_feels"}
print(cycle_data.shape)
cycle_data.rename(columns=col_mappings, inplace=True)
cycle_data.head()


# %%
# Split timestamps
import datetime

# 2015-01-04 00:00:00
cycle_data["timestamp_obj"] = cycle_data["timestamp"].apply(
    lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
)

cycle_data["year"] = cycle_data["timestamp_obj"].apply(lambda x: x.year)
cycle_data["month"] = cycle_data["timestamp_obj"].apply(lambda x: x.month)
cycle_data["day"] = cycle_data["timestamp_obj"].apply(lambda x: x.day)
cycle_data["hour"] = cycle_data["timestamp_obj"].apply(lambda x: x.hour)
cycle_data["week_day"] = cycle_data["timestamp_obj"].apply(lambda x: x.weekday() + 1)
cycle_data.head()


# %%
# Aggregate by day and plot against time
cycle_day_data = (
    cycle_data[["year", "month", "day", "week_day", "count"]]
    .groupby(by=["year", "month", "day", "week_day"])
    .sum()
    .reset_index()
)
cycle_day_data["datetime"] = cycle_day_data.apply(
    func=lambda x: datetime.date(x["year"], x["month"], x["day"]), axis=1
)
cycle_day_data = cycle_day_data[
    ["datetime", "count", "year", "month", "day", "week_day"]
]
cycle_day_data.head()

# %% [markdown]
# ## Things to explore
# * Correlation of temperature/bad weather against counts
#   * Higher correlation on weekends/holidays
# * More journeys at rush hour
# * Journeys increasing with time?

# %%
# More journeys per hour on weekends?
count_data = [
    cycle_data[cycle_data["is_weekend"] == 0]["count"],
    cycle_data[cycle_data["is_weekend"] == 1]["count"],
]
mean_counts = [np.mean(data) for data in count_data]
plt.hist(
    count_data,
    bins=30,
    range=(0, 5000),
    label=[f"Weekday, µ: {mean_counts[0]:.0f}", f"Weekend, µ: {mean_counts[1]:.0f}"],
)
plt.tight_layout()
plt.legend(loc="upper right")
plt.xlabel("Number of trips/hour")
plt.ylabel("Freq")
plt.show()

# By day
count_data = [
    cycle_day_data[~cycle_day_data["week_day"].isin([6, 7])]["count"],
    cycle_day_data[cycle_day_data["week_day"].isin([6, 7])]["count"],
]
mean_counts = [np.mean(data) for data in count_data]
plt.hist(
    count_data,
    bins=30,
    label=[f"Weekday, µ: {mean_counts[0]:.0f}", f"Weekend, µ: {mean_counts[1]:.0f}"],
)
plt.tight_layout()
plt.legend(loc="upper right")
plt.xlabel("Number of trips/day")
plt.ylabel("Freq")
plt.show()


# %%
# Journeys increasing with time?
fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
ax = fig.subplots()
# ax.scatter(cycle_day_data["datetime"], cycle_day_data["count"], alpha=0.2)
ax.scatter("datetime", "count", data=cycle_day_data, alpha=0.2)
plt.xlabel("Date")
plt.ylabel("Number of trips/day")

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
# plt.savefig("TFLCycles/images/against_time.png")
plt.show()

# Maybe better to show change in totals per week year on year


# %% [markdown]
# # Journeys by hour
# The distribution is different weekday to weekend

# %%
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
sns.boxplot(x="hour", y="count", hue="is_weekend", data=cycle_data)
# plt.tight_layout()
plt.xlabel("Hour")
plt.ylabel("Number of trips/hour")
plt.title("Weekend has different dist to weekday")
# plt.savefig("TFLCycles/images/journeys_per_hour_boxplot.png")
plt.show()


# %%
# Against week day
# Monday seems least popular work day to cycle, weekend less popular
plt.figure(num=None, figsize=(16, 6), dpi=80, facecolor="w", edgecolor="k")
sns.boxplot(x="week_day", y="count", data=cycle_day_data)
plt.tight_layout()
plt.xlabel("Day of week")
plt.ylabel("Number of trips/day")
# plt.savefig("TFLCycles/images/journeys_per_week.png")
plt.show()


# %%
# Against month
# Winter months are less popular
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
sns.boxplot(x="month", y="count", data=cycle_day_data)
plt.tight_layout()
plt.xlabel("Month")
plt.ylabel("Number of trips/day")
# plt.savefig("TFLCycles/images/journeys_per_month.png")
plt.show()

# %% [markdown]
# # Does time of day change through out the year?
# More later journeys in summer?
# When normalised over the day - a higher proportion of journeys are made later in the evening

# %%
# Heat map against month/hour
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
agr_counts = (
    cycle_data[["month", "hour", "count"]].groupby(by=["month", "hour"], axis=0).mean()
)
agr_counts_pivot = agr_counts.reset_index().pivot(
    index="month", columns="hour", values="count"
)
sns.heatmap(agr_counts_pivot)
plt.title("Mean journeys per hour")
plt.xlabel("Hour")
plt.ylabel("Month")
# plt.savefig("TFLCycles/images/journeys_per_hour_month.png")
plt.show()

# Normalise over the day - higher proportion of journeys made later in the evening
agr_counts_norm = agr_counts.groupby("month").transform(lambda x: (x / x.sum()))
agr_counts_norm_pivot = agr_counts_norm.reset_index().pivot(
    index="month", columns="hour", values="count"
)
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
sns.heatmap(agr_counts_norm_pivot)
plt.title("% journeys per hour")
plt.xlabel("Hour")
plt.ylabel("Month")
# plt.savefig("TFLCycles/images/journeys_per_hour_month_prop.png")
plt.show()

# %% [markdown]
# # Does time of day change through out the week?
# Friday has a more flat distribution of journey times and a lower early evening proportion and a higher later evening proportion
# Weekends dont exhibit a morning peak but longer evening tails

# %%
# Heat map against day/hour
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
agr_counts = (
    cycle_data[["week_day", "hour", "count"]]
    .groupby(by=["week_day", "hour"], axis=0)
    .mean()
)
agr_counts_pivot = agr_counts.reset_index().pivot(
    index="week_day", columns="hour", values="count"
)
sns.heatmap(agr_counts_pivot)
plt.title("Mean journeys per hour")
plt.xlabel("Hour")
plt.ylabel("Week day")
# plt.savefig("TFLCycles/images/journeys_per_hour.png")
plt.show()

# Normalise over the day - higher proportion of journeys made later in the evening
agr_counts_norm = agr_counts.groupby("week_day").transform(lambda x: (x / x.sum()))
agr_counts_norm_pivot = agr_counts_norm.reset_index().pivot(
    index="week_day", columns="hour", values="count"
)
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
sns.heatmap(agr_counts_norm_pivot)
plt.title("% journeys per hour")
plt.xlabel("Hour")
plt.ylabel("Week day")
# plt.savefig("TFLCycles/images/journeys_per_hour_prop.png")
plt.show()

# %% [markdown]
# # Relation to weather
# Temperature/humdity/wind speed affects counts?
#
# 'Real feel' temperature is very similar to temperature other than low temperatures so only using temp_feels for now
# No noticable difference against weekday/weekend - isnt clear whether on weekdays people endure the weather regardless
# High temperatures = more journeys
#
#
# "weather_code" category description:
# 1 = Clear ; mostly clear but have some values with haze/fog/patches of fog/ fog in vicinity
# 2 = scattered clouds / few clouds
# 3 = Broken clouds
# 4 = Cloudy
# 7 = Rain/ light Rain shower/ Light rain
# 10 = rain with thunderstorm
# 26 = snowfall
# 94 = Freezing Fog

# %%
# temperature vs temp_feels
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
sns.scatterplot(x="temperature", y="temp_feels", data=cycle_data, alpha=0.5)
plt.tight_layout()
plt.show()

# No clear pattern when splitting by weekend effects
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
sns.scatterplot(x="temp_feels", y="count", hue="is_weekend", data=cycle_data, alpha=0.5)
plt.tight_layout()
plt.ylabel("Number of trips/hour")
plt.show()

# Wind speed?
# High speed fewer journies
# Low speed too - other confounding factors?

# High temperatures are correlat
group_size = 2.5
cycle_data["temp_feels_rn"] = (
    cycle_data["temp_feels"] / group_size
).round() * group_size
# .apply(lambda x: )
plt.figure(num=None, figsize=(16, 6), dpi=80, facecolor="w", edgecolor="k")
sns.boxplot(x="temp_feels_rn", y="count", data=cycle_data)
plt.tight_layout()
# plt.xlabel('Month')
plt.ylabel("Number of trips/hour")
plt.show()


# %%
# Find average weather for each day
# Limit to day time only?
modefcn = lambda x: pd.Series.mode(x)[0]
idx_cols = ["year", "month", "day", "week_day"]
f = {
    "count": "sum",
    "temp_feels": "mean",
    "wind_speed": "mean",
    "hum": "mean",
    "weather_code": pd.Series.mode,
}
cycle_day_data_agg = (
    cycle_data[idx_cols + list(f.keys())].groupby(by=idx_cols).agg(f).reset_index()
)
cycle_day_data_agg["datetime"] = cycle_day_data_agg.apply(
    func=lambda x: datetime.date(x["year"], x["month"], x["day"]), axis=1
)
cycle_day_data_agg["is_weekend"] = cycle_day_data_agg["week_day"].isin([6, 7])

join_cols = idx_cols + ["datetime", "count"]
cycle_day_data = cycle_day_data.join(cycle_day_data_agg, rsuffix="_right")
cycle_day_data.drop(columns=[col + "_right" for col in join_cols], inplace=True)
cycle_day_data.head()

# %% Fix mode array issues
def modefcn(x):
    # take first value - introduces some bias
    if isinstance(x, np.ndarray):
        return x[0]
    else:
        return x


cycle_day_data["weather_code"] = cycle_day_data["weather_code"].apply(modefcn)

# %%
# Temperature vs wind speed? -  no strong correlation
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
sns.scatterplot(x="temp_feels", y="wind_speed", data=cycle_day_data)
plt.tight_layout()
plt.show()

# %%
# Wind speed?
# High speed seems fewer journeys
# Insufficient to say wind  speed affects number of journeys?

# Scatter doesnt show much - negative relationship?
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
# sns.boxplot(x="wind_speed", y="count", data=cycle_day_data)
sns.scatterplot(x="wind_speed", y="count", data=cycle_day_data)
plt.tight_layout()
# plt.xlabel('Month')
plt.ylabel("Number of trips")
plt.show()

# Use grouped box plot instead
group_size = 2.0
cycle_day_data["wind_speed_rnd"] = cycle_day_data["wind_speed"] / group_size
cycle_day_data["wind_speed_rnd"] = cycle_day_data["wind_speed_rnd"].round() * group_size
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
sns.boxplot(x="wind_speed_rnd", y="count", data=cycle_day_data)
plt.tight_layout()
plt.ylabel("Number of trips/day")
plt.xlabel("Wind speed")
plt.show()

# Regress counts on wind speeds

# %% Temperature
# Strong positive correlation, likely confounds the seasonal trend
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
sns.scatterplot(
    x="temp_feels", y="count", hue="is_weekend", data=cycle_day_data, alpha=0.5
)
plt.tight_layout()
plt.ylabel("Number of trips/day")
plt.show()

# %% Humidity
# Strong negative correlation, likely confounds the seasonal trend, lower humidity with higher temperatures
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
sns.scatterplot(x="hum", y="count", hue="is_weekend", data=cycle_day_data, alpha=0.5)
plt.tight_layout()
plt.ylabel("Number of trips/day")
plt.show()

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
sns.scatterplot(
    x="temp_feels", y="hum", hue="is_weekend", data=cycle_day_data, alpha=0.5
)
plt.tight_layout()
plt.show()

# %%
# Against weather type
# No one cycles in snow...
# Fewer cycles in the rain

cycle_day_data["weather_code_label"] = cycle_day_data["weather_code"].replace(
    {
        1: "Clear",
        2: "Scattered clouds",
        3: "Broken clouds",
        4: "Cloudy",
        7: "Rain",
        26: "Snowfall",
    }
)

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
sns.boxplot(
    x="weather_code_label",
    y="count",
    data=cycle_day_data,
    order=["Clear", "Scattered clouds", "Broken clouds", "Cloudy", "Rain", "Snowfall"],
)
plt.tight_layout()
plt.ylabel("Number of trips")
plt.savefig("TFLCycles/images/weather_codes.png")
plt.show()


print(
    """
"weather_code" category description:\n
1 = Clear ; mostly clear but have some values with haze/fog/patches of fog/ fog in vicinity  \n
2 = scattered clouds / few clouds  \n
3 = Broken clouds  \n
4 = Cloudy  \n
7 = Rain/ light Rain shower/ Light rain  \n
10 = rain with thunderstorm  \n
26 = snowfall  \n
94 = Freezing Fog \n  
"""
)

# %% Export CSV
cycle_day_data.to_csv("TFLCycles/data/london_merged_processed.csv", index=False)


# %%
