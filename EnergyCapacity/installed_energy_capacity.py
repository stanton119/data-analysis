# %%

import numpy as np
import pandas as pd
import os
import pathlib

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
# import hvplot.pandas
import holoviews as hv

hv.extension("bokeh")

# %% Load data
area_type = "CTY"
output_path = pathlib.Path(os.getcwd()) / "data" / area_type
country_names = [
    f.replace(".parquet", "")
    for f in os.listdir(output_path)
    if ".parquet" in f
]
country_names = sorted(country_names)

cap_tables = []
for country in country_names:
    cap_tables.append(pd.read_parquet(output_path / (country + ".parquet")))

# %%
energy_groups = {
    "Wind Onshore": "Wind",
    "Wind Offshore": "Wind",
    "Hydro Pumped Storage": "Hydro",
    "Hydro Run-of-river and poundage": "Hydro",
}

energy_types = []
for table in cap_tables:
    energy_types += list(table.columns)
energy_types = list(set(energy_types))

fossil_types = [
    energy_type for energy_type in energy_types if "Fossil" in energy_type
]
non_fossil_types = [
    energy_type for energy_type in energy_types if "Fossil" not in energy_type
]
non_fossil_types.remove("Total Grand capacity")

# %%
def prep_table(df):
    temp = df.copy()
    # drop missing types
    temp2 = temp.isna().all(axis=0)
    temp = temp.drop(
        columns=list(temp2[temp2].index) + ["Total Grand capacity"]
    )
    if 0:
        sort_idx = temp.iloc[-1].sort_values(ascending=False)
        sort_idx = sort_idx.index
        temp = temp.loc[:, sort_idx]

        temp.rename(columns=energy_groups)
    return temp


def sum_renewables_table(df):
    df_sum = pd.DataFrame(index=df.index)
    df_sum["Fossil"] = df.loc[:, fossil_types].sum(axis=1)
    df_sum["Renewables"] = df.loc[:, non_fossil_types].sum(axis=1)
    df_sum[df_sum == 0] = np.nan
    return df_sum


def line_plot(df, ax, title):
    df.plot(ax=ax)
    ax.set_title(title)
    # ax.set_ylim(0, df.max().max() * 1.1)
    ax.set_ylim(0, 150000)
    # ax.set_xlim(2015, 2020)

# %% Grid plot all countries, same scale
subplot_ncols = 4
subplot_nrows = int(np.ceil(len(cap_tables) / subplot_ncols))
fig, ax = plt.subplots(
    ncols=subplot_ncols, nrows=subplot_nrows, figsize=(12, 4 * subplot_nrows)
)
max_caps = np.array(
    [table["Total Grand capacity"].mean() for table in cap_tables]
)
max_caps[np.isnan(max_caps)] = -1
sort_idx = np.flip(np.argsort(max_caps))
for idx, table_idx in enumerate(sort_idx):
    try:
        line_plot(
            sum_renewables_table(cap_tables[table_idx]),
            ax[idx // subplot_ncols, idx % subplot_ncols],
            country_names[table_idx],
        )
    except:
        pass


# %% Plot % of renewables
# Only biggest countries
fig, ax = plt.subplots(
    figsize=(12, 12)
)
max_caps = np.array(
    [table["Total Grand capacity"].mean() for table in cap_tables]
)
max_caps[np.isnan(max_caps)] = -1
sort_idx = np.flip(np.argsort(max_caps))
for idx, table_idx in enumerate(sort_idx[:12]):
    try:
        temp = sum_renewables_table(cap_tables[table_idx])
        temp = temp['Renewables']/temp.sum(axis=1)
        temp.plot(ax=ax, label=country_names[table_idx])
    except:
        pass
ax.set_xlabel('Year')
ax.set_ylabel('% renewables')
ax.set_ylim(0, 1)
ax.set_title('Installed renewables / total')
plt.legend()

# %% Plot individual countries
idx = -1
cap_tables[idx].transpose()
country_names[idx]
# %%
temp = cap_tables[1]
prep_table(temp).plot(kind="area")
prep_table(temp).hvplot()

# %%
sort_idx = np.argsort(
    [table["Total Grand capacity"].mean() for table in cap_tables]
)

sum_renewables_table(cap_tables[-3]).max().max()
sum_renewables_table(cap_tables[-3]).plot(kind="area")
sum_renewables_table(cap_tables[-3]).plot()

# %%
fig = prep_table(table).hvplot(grid=True)
fig

# %%
grid_style = {
    "grid_line_color": "black",
    "grid_line_width": 1.5,
    "ygrid_bounds": (0.3, 0.7),
    "minor_xgrid_line_color": "lightgray",
    "xgrid_line_dash": [4, 4],
}

hv.Points(np.random.rand(10, 2)).opts(
    gridstyle=grid_style, show_grid=True, size=5, width=600
)

# %%
import plotly.graph_objects as go

x = ["Winter", "Spring", "Summer", "Fall"]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=x,
        y=[40, 60, 40, 10],
        hoverinfo="x+y",
        mode="lines",
        line=dict(width=0.5, color="rgb(131, 90, 241)"),
        # stackgroup='one' # define stack group
    )
)
fig.add_trace(
    go.Scatter(
        x=x,
        y=[20, 10, 10, 60],
        hoverinfo="x+y",
        mode="lines",
        line=dict(width=0.5, color="rgb(111, 231, 219)"),
        # stackgroup='one'
    )
)
fig.add_trace(
    go.Scatter(
        x=x,
        y=[40, 30, 50, 30],
        hoverinfo="x+y",
        mode="lines",
        line=dict(width=0.5, color="rgb(184, 247, 212)"),
        # stackgroup='one'
    )
)

fig.update_layout(yaxis_range=(0, 100))
fig.show()

# %%
# Only biggest countries
fig, ax = plt.subplots(figsize=(12, 12))
max_caps = np.array(
    [table["Total Grand capacity"].mean() for table in cap_tables]
)
max_caps[np.isnan(max_caps)] = -1
sort_idx = np.flip(np.argsort(max_caps))
for idx, table_idx in enumerate(sort_idx[:10]):
    try:
        temp = sum_renewables_table(cap_tables[table_idx])
        temp = temp['Renewables']/temp.sum(axis=1)
        temp.plot(ax=ax, label=country_names[table_idx])
    except:
        pass
ax.set_xlabel('Year')
ax.set_ylabel('% renewables')
ax.set_ylim(0, 1)
ax.set_title('Installed renewables / total')
plt.legend()
plt.show()