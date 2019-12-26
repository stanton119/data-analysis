"""
Ideas
Map single bike over the day
Plot all in/out from one station
Journeys from central shorter than more outside?
Predict number of bikes available
Find longest journey - ride london?

"""

#%% Imports
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

#%% Load data
# https://cycling.data.tfl.gov.uk
cycle_journeys = pd.read_csv("TFLCycles/103JourneyDataExtract28Mar2018-03Apr2018.csv")

#%%
for column in cycle_journeys.columns:
    cycle_journeys.columns

#%% Combine multiple files

#%% Create station ID lookup table
station_id1 = cycle_journeys[["EndStation Id", "EndStation Name"]]
station_id1.columns = ["StationID", "StationName"]
station_id2 = cycle_journeys[["StartStation Id", "StartStation Name"]]
station_id2.columns = ["StationID", "StationName"]

station_ids = pd.concat([station_id1, station_id2], axis=0)
del station_id1, station_id2
station_ids.drop_duplicates(inplace=True)
station_ids.sort_values(by="StationID", inplace=True)
station_ids.head()

#%% Find data from stationID
ids_1 = station_ids[station_ids["StationName"].str.contains("Grant", case=False)][
    "StationID"
].values
ids_2 = station_ids[
    station_ids["StationName"].str.contains("peter's terrace", case=False)
]["StationID"].values

print(ids_1)
print(ids_2)

#%%
filter = np.logical_or(
    cycle_journeys["EndStation Id"].isin(ids_1),
    cycle_journeys["StartStation Id"].isin(ids_1),
)
cycle_journeys[filter]

#%% Find journey distance
cycle_journeys["EndDateTime"] = pd.to_datetime(
    cycle_journeys["End Date"], format="%d/%m/%Y %H:%M"
)
cycle_journeys["StartDateTime"] = pd.to_datetime(
    cycle_journeys["Start Date"], format="%d/%m/%Y %H:%M"
)
cycle_journeys["JourneyLength"] = (
    cycle_journeys["EndDateTime"] - cycle_journeys["StartDateTime"]
)
cycle_journeys["JourneyLength"] = cycle_journeys["JourneyLength"].apply(
    lambda x: x.seconds / 60
)


#%% Histogram of journeys under 2 hours
filter = cycle_journeys["JourneyLength"] < 60 * 2
plt.hist(cycle_journeys["JourneyLength"][filter], bins=200)
plt.show()

#%%
ax = cycle_journeys["JourneyLength"][filter].plot.kde()
plt.show()

#%%
