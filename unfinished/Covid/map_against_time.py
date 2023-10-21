"""
Plotting cases against time on a map

Data source:
https://coronavirus.data.gov.uk/details/download
https://coronavirus.data.gov.uk/details/developers-guide/main-api#schema
https://coronavirus.data.gov.uk/details/developers-guide/generic-api

https://osdatahub.os.uk/downloads/open/BoundaryLine?_ga=2.261733064.479182950.1630362176-723482396.1630362176
"""

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-whitegrid")

# %%
df = pd.read_csv('/Users/Rich/Downloads/ltla_2021-06-01.csv')
df

# %%
df['areaName'].value_counts()

df['areaNameID'] = df['areaName'].astype('category').cat.code

df['date'] = pd.to_datetime(df['date'])
df['dateInt'] = df['date'].astype(int)


# %%
fig, ax = plt.subplots(figsize=(12, 6))
ax = sns.heatmap(df[['areaNameID','dateInt','cumCasesByPublishDate']], cmap="plasma", ax=ax)
ax.set_title("Minutes Played")
plt.show()

# %%
# map to lat/long
df_locations = pd.read_csv('/Users/Rich/Downloads/Lower_Layer_Super_Output_Areas_(December_2001)_Population_Weighted_Centroids.csv')
df_locations

df['areaCode'][0]
df_locations.loc[df_locations['lsoa01cd']==df['areaCode'][0]]


# %%
import pandas as pd
quakes = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv')

import plotly.graph_objects as go
fig = go.Figure(go.Densitymapbox(lat=quakes.Latitude, lon=quakes.Longitude, z=quakes.Magnitude,
                                 radius=10))
fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=180)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# %%
import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

# %%
df = pd.read_excel(
    pathlib.Path(__file__).parent / "ons.xlsx", sheet_name="Table 3", 
    header=14, usecols='A:B,R,AH') 


df = df.iloc[:23, :6]
df.drop(index=[0, 1, 2], inplace=True)
cols = list(df.columns)
cols[0] = 'Age'
df.columns = cols
df['Age'] = df['Age'].apply(lambda x: x[:2])
df.set_index('Age', inplace=True)
df['Total deaths'] = df.sum(axis=1)
df['Cumulative total deaths'] = df['Total deaths'].cumsum()

# %%
fig, ax = plt.subplots(figsize=(10,6))
df.plot(y=['Cumulative total deaths', 'Total deaths'], ax=ax)
