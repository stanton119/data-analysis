# %% [markdown]
"""
Functions used to preprocess the TFL cycle data
"""


# %%
import os
import datetime
import numpy as np
import pandas as pd


# %%
def load_tfl_csv() -> pd.DataFrame:
    # Fetch data into dataframe
    dathpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    filepath = os.path.join(dathpath, "london_merged.csv")
    cycle_data = pd.read_csv(filepath)
    return cycle_data


# %%
def change_column_names(cycle_data: pd.DataFrame) -> pd.DataFrame:
    col_mappings = {"cnt": "count", "t1": "temp", "t2": "temp_feels"}
    return cycle_data.rename(columns=col_mappings)


# %%
def convert_to_timestamp_objects(cycle_data: pd.DataFrame) -> pd.DataFrame:
    # 2015-01-04 00:00:00
    cycle_data["timestamp_obj"] = cycle_data["timestamp"].apply(
        lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    )

    cycle_data["year"] = cycle_data["timestamp_obj"].apply(lambda x: x.year)
    cycle_data["month"] = cycle_data["timestamp_obj"].apply(lambda x: x.month)
    cycle_data["day"] = cycle_data["timestamp_obj"].apply(lambda x: x.day)
    cycle_data["hour"] = cycle_data["timestamp_obj"].apply(lambda x: x.hour)
    cycle_data["week_day"] = cycle_data["timestamp_obj"].apply(
        lambda x: x.weekday() + 1
    )

    return cycle_data


# %%
# Aggregate count data by day
if 0:
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


# %%
def aggregate_data_over_each_day(cycle_data: pd.DataFrame) -> pd.DataFrame:
    # Find average weather for each day
    # Limit to day time only?
    idx_cols = ["year", "month", "day", "week_day"]
    f = {
        "count": "sum",
        "temp_feels": "mean",
        "wind_speed": "mean",
        "hum": "mean",
        "weather_code": pd.Series.mode,
    }
    cycle_day_data = cycle_data[idx_cols + list(f.keys())].groupby(by=idx_cols).agg(f)
    cycle_day_data["is_weekend"] = cycle_day_data.index.get_level_values(
        "week_day"
    ).isin([6, 7])

    cycle_day_data = fix_weather_mode_array_issues(cycle_day_data)

    return cycle_day_data


def fix_weather_mode_array_issues(cycle_day_data: pd.DataFrame) -> pd.DataFrame:
    def modefcn(x):
        # take first value - introduces some bias
        if isinstance(x, np.ndarray):
            return x[0]
        else:
            return x

    cycle_day_data["weather_code"] = cycle_day_data["weather_code"].apply(modefcn)
    return cycle_day_data


# %%
def weather_type_to_str(cycle_day_data: pd.DataFrame) -> pd.DataFrame:
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
    return cycle_day_data


# %% Export parquet
def export_parquet(cycle_day_data: pd.DataFrame):
    dathpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    cycle_day_data.to_parquet(os.path.join(dathpath, "london_merged_processed.pq"))
