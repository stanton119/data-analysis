# Installed Energy Capacity in Europe
Here we will find some data to see what the state of renewable energy production is within Europe.
The data is collected from:
https://transparency.entsoe.eu/generation/r2/installedGenerationCapacityAggregation/show


```python
import numpy as np
import pandas as pd
import os
import pathlib

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
```

The data is collected via the script `scrape_data.py`. Here we load it all back in. For each country we have a dataframe of energy production capacities.


```python
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
```

### Quick inspection
We can see the contents of the dataframe for Germany. Each column is the type of the installed energy production:


```python
idx = country_names.index('Germany (DE)')
cap_tables[idx]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Production Type</th>
      <th>Fossil Peat</th>
      <th>Nuclear</th>
      <th>Fossil Hard coal</th>
      <th>Wind Onshore</th>
      <th>Fossil Brown coal/Lignite</th>
      <th>Geothermal</th>
      <th>Hydro Run-of-river and poundage</th>
      <th>Hydro Water Reservoir</th>
      <th>Wind Offshore</th>
      <th>Hydro Pumped Storage</th>
      <th>...</th>
      <th>Solar</th>
      <th>Fossil Oil shale</th>
      <th>Waste</th>
      <th>Fossil Gas</th>
      <th>Fossil Coal-derived gas</th>
      <th>Fossil Oil</th>
      <th>Marine</th>
      <th>Other</th>
      <th>Biomass</th>
      <th>Total Grand capacity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015</th>
      <td>NaN</td>
      <td>12068.0</td>
      <td>26190.0</td>
      <td>37701.0</td>
      <td>21160.0</td>
      <td>34.0</td>
      <td>3989.0</td>
      <td>1518.0</td>
      <td>993.0</td>
      <td>8699.0</td>
      <td>...</td>
      <td>37271.0</td>
      <td>NaN</td>
      <td>1685.0</td>
      <td>31734.0</td>
      <td>NaN</td>
      <td>4532.0</td>
      <td>NaN</td>
      <td>1220.0</td>
      <td>6808.0</td>
      <td>196051.0</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>NaN</td>
      <td>10793.0</td>
      <td>26264.0</td>
      <td>41168.0</td>
      <td>21062.0</td>
      <td>34.0</td>
      <td>3996.0</td>
      <td>1518.0</td>
      <td>3283.0</td>
      <td>8699.0</td>
      <td>...</td>
      <td>38686.0</td>
      <td>NaN</td>
      <td>1685.0</td>
      <td>32398.0</td>
      <td>NaN</td>
      <td>4605.0</td>
      <td>NaN</td>
      <td>1286.0</td>
      <td>6815.0</td>
      <td>202803.0</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>NaN</td>
      <td>10793.0</td>
      <td>27437.0</td>
      <td>47042.0</td>
      <td>21262.0</td>
      <td>40.0</td>
      <td>4007.0</td>
      <td>1439.0</td>
      <td>4131.0</td>
      <td>8894.0</td>
      <td>...</td>
      <td>40834.0</td>
      <td>NaN</td>
      <td>1685.0</td>
      <td>32627.0</td>
      <td>NaN</td>
      <td>4614.0</td>
      <td>NaN</td>
      <td>1421.0</td>
      <td>7080.0</td>
      <td>213816.0</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>NaN</td>
      <td>9516.0</td>
      <td>25035.0</td>
      <td>51633.0</td>
      <td>21275.0</td>
      <td>38.0</td>
      <td>3860.0</td>
      <td>1440.0</td>
      <td>5051.0</td>
      <td>8918.0</td>
      <td>...</td>
      <td>42804.0</td>
      <td>NaN</td>
      <td>1686.0</td>
      <td>31361.0</td>
      <td>NaN</td>
      <td>4271.0</td>
      <td>NaN</td>
      <td>1418.0</td>
      <td>7396.0</td>
      <td>216198.0</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>NaN</td>
      <td>9516.0</td>
      <td>25293.0</td>
      <td>52792.0</td>
      <td>21205.0</td>
      <td>42.0</td>
      <td>3983.0</td>
      <td>1298.0</td>
      <td>6393.0</td>
      <td>9422.0</td>
      <td>...</td>
      <td>45299.0</td>
      <td>NaN</td>
      <td>1686.0</td>
      <td>31664.0</td>
      <td>NaN</td>
      <td>4356.0</td>
      <td>NaN</td>
      <td>1235.0</td>
      <td>7752.0</td>
      <td>222381.0</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>NaN</td>
      <td>8114.0</td>
      <td>22458.0</td>
      <td>53405.0</td>
      <td>21067.0</td>
      <td>42.0</td>
      <td>3958.0</td>
      <td>1298.0</td>
      <td>7709.0</td>
      <td>9422.0</td>
      <td>...</td>
      <td>46471.0</td>
      <td>NaN</td>
      <td>1661.0</td>
      <td>31712.0</td>
      <td>0.0</td>
      <td>4373.0</td>
      <td>NaN</td>
      <td>1558.0</td>
      <td>7855.0</td>
      <td>221584.0</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 21 columns</p>
</div>



### Largest capacity country
Which countries have the most energy capacity?


```python
max_caps = np.array(
    [table["Total Grand capacity"].mean() for table in cap_tables]
)
max_cap = pd.DataFrame()
max_cap['Installed production (MW)'] = np.round(max_caps)
max_cap.index = country_names
max_cap = max_cap.sort_values(by='Installed production (MW)', ascending=False)
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlabel('Countries')
ax.set_ylabel('Installed production (MW)')
max_cap.plot(use_index=False, ax=ax, style='x-')
max_cap.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Installed production (MW)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Germany (DE)</th>
      <td>212139.0</td>
    </tr>
    <tr>
      <th>France (FR)</th>
      <td>126668.0</td>
    </tr>
    <tr>
      <th>Spain (ES)</th>
      <td>105300.0</td>
    </tr>
    <tr>
      <th>Italy (IT)</th>
      <td>96132.0</td>
    </tr>
    <tr>
      <th>United Kingdom (UK)</th>
      <td>68215.0</td>
    </tr>
    <tr>
      <th>Poland (PL)</th>
      <td>39418.0</td>
    </tr>
    <tr>
      <th>Sweden (SE)</th>
      <td>38837.0</td>
    </tr>
    <tr>
      <th>Netherlands (NL)</th>
      <td>32177.0</td>
    </tr>
    <tr>
      <th>Norway (NO)</th>
      <td>30734.0</td>
    </tr>
    <tr>
      <th>Belgium (BE)</th>
      <td>21894.0</td>
    </tr>
  </tbody>
</table>
</div>




![svg](installed_energy_capacity_files/installed_energy_capacity_7_1.svg)


Germany in this dataset is much larger than other countries, and there is a fast fall off with the small countries.

### Renewables breakdown
We can plot all the countries summarised by a breakdown of renewables/fossil fuels. The countries are ordered by total installed energy production capacity.


```python
# Split energy into renewables and fossil fuels
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

# define helper functions
def sum_renewables_table(df):
    df_sum = pd.DataFrame(index=df.index)
    df_sum["Fossil"] = df.loc[:, fossil_types].sum(axis=1)
    df_sum["Renewables"] = df.loc[:, non_fossil_types].sum(axis=1)
    df_sum[df_sum == 0] = np.nan
    return df_sum

def line_plot(df, ax, title):
    df.plot(ax=ax)
    ax.set_title(title)
    ax.set_ylim(0, df.max().max() * 1.1)
```


```python
subplot_ncols = 4
subplot_nrows = 4 #int(np.ceil(len(cap_tables) / subplot_ncols))
fig, ax = plt.subplots(
    ncols=subplot_ncols, nrows=subplot_nrows, figsize=(12, 4 * subplot_nrows)
)
max_caps = np.array(
    [table["Total Grand capacity"].mean() for table in cap_tables]
)
max_caps[np.isnan(max_caps)] = -1
sort_idx = np.flip(np.argsort(max_caps))
for idx, table_idx in enumerate(sort_idx[:16]):
    try:
        line_plot(
            sum_renewables_table(cap_tables[table_idx]),
            ax[idx // subplot_ncols, idx % subplot_ncols],
            country_names[table_idx],
        )
    except:
        pass
```


![svg](installed_energy_capacity_files/installed_energy_capacity_10_0.svg)


Of the bigger countries - new energy capacity in Germany and France have been moving in the right direction, but the fossils have not reduced much. Italy has been moving in the wrong direction, installing more fossil fuels. The UK has also increased it capacity of fossil fuels.

## Production type breakdown
We can plot all production types together to see what has changed:


```python
energy_groups = {
    "Biomass": "Other renewable",
    "Fossil Brown coal/Lignite": "Fossil Other",
    "Fossil Coal-derived gas": "Fossil Gas",
    "Fossil Gas": "Fossil Gas",
    "Fossil Hard coal": "Fossil Coal",
    "Fossil Oil": "Fossil Other",
    "Fossil Oil shale": "Fossil Other",
    "Fossil Peat": "Fossil Other",
    "Geothermal": "Other renewable",
    "Hydro Pumped Storage": "Hydro",
    "Hydro Run-of-river and poundage": "Hydro",
    "Hydro Water Reservoir": "Hydro",
    "Marine": "Other Renewable",
    "Nuclear": "Nuclear",
    "Other": "Other",
    "Other Renewable": "Other Renewable",
    "Solar": "Solar",
    "Waste": "Other Renewable",
    "Wind Offshore": "Wind",
    "Wind Onshore": "Wind",
}

def prep_table(df):
    temp = df.copy()
    # drop missing types
    temp2 = temp.isna().all(axis=0)
    temp = temp.drop(
        columns=list(temp2[temp2].index) + ["Total Grand capacity"]
    )

    # rename and combine columns
    temp = temp.rename(columns=energy_groups)
    for key in list(set(energy_groups.values())):
        try:
            temp2 = temp.loc[:, [key]].sum(axis=1)
            temp.drop(columns=key, inplace=True)
            temp[key] = temp2
        except:
            temp[key] = 0

    return temp
```


```python
subplot_ncols = 4
subplot_nrows = 4
fig, ax = plt.subplots(
    ncols=subplot_ncols, nrows=subplot_nrows, figsize=(12, 4 * subplot_nrows)
)
max_caps = np.array(
    [table["Total Grand capacity"].mean() for table in cap_tables]
)
max_caps[np.isnan(max_caps)] = -1
sort_idx = np.flip(np.argsort(max_caps))
for idx, table_idx in enumerate(sort_idx[:16]):
    prep_table(cap_tables[table_idx]).plot(kind="area", ax=ax[idx // subplot_ncols, idx % subplot_ncols], title=country_names[table_idx])
    ax[idx // subplot_ncols, idx % subplot_ncols].legend_.remove()


handles, labels = ax[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='left')

plt.show()
```


![svg](installed_energy_capacity_files/installed_energy_capacity_14_0.svg)


Germany has installed a log more wind and solar power which has increased in renewables. France has increased in wind power.

The UK and Italy have both increased the uptake of fossil gas power.

Much of Scandinavia has a high percentage of renewables, mainly hydro.


Some of these changes point to issues with the data. The UK has no nuclear output listed and has no solar before 2017.
