# %%
import numpy as np
import pandas as pd
import requests
import os
import pathlib
from bs4 import BeautifulSoup

# %%
area_type = 'CTY'   # BZN - for bidding zone

# %% List countryies and url codes
base_url=f"https://transparency.entsoe.eu/generation/r2/installedGenerationCapacityAggregation/show?name=&defaultValue=false&viewType=TABLE&areaType={area_type}"
r = requests.get(base_url)
country_list_html = BeautifulSoup(r.text, 'html.parser')

countries = {}
for country in country_list_html.find_all(attrs={'class': "dv-filter-hierarchic-wrapper"}):
    country_name = country.find(name='label').string
    countries[country_name] = []

    for label in country.find_all(attrs={'class': "dv-filter-checkbox"})[1:]:
        countries[country_name].append(label.find('input').get('value'))

# %%
cap_tables = []
output_path = pathlib.Path(os.getcwd()) / 'data' / area_type
output_path.mkdir(parents=True, exist_ok=True)
for country in countries:
    print(f"processing: {country}")
    value = countries[country][0]
    country_cap_url = f"https://transparency.entsoe.eu/generation/r2/installedGenerationCapacityAggregation/show?name=&defaultValue=false&viewType=TABLE&areaType={area_type}&atch=false&dateTime.dateTime=01.01.2015+00:00|UTC|YEAR&dateTime.endDateTime=01.01.2020+00:00|UTC|YEAR&area.values={value}&productionType.values=B01&productionType.values=B02&productionType.values=B03&productionType.values=B04&productionType.values=B05&productionType.values=B06&productionType.values=B07&productionType.values=B08&productionType.values=B09&productionType.values=B10&productionType.values=B11&productionType.values=B12&productionType.values=B13&productionType.values=B14&productionType.values=B20&productionType.values=B15&productionType.values=B16&productionType.values=B17&productionType.values=B18&productionType.values=B19"

    r = requests.get(country_cap_url)
    cap_html = BeautifulSoup(r.text, 'html.parser')

    table = pd.read_html(str(cap_html.find(id="dv-data-table").table), flavor='bs4')[0]
    table.columns = table.columns.get_level_values(0)
    table.set_index('Production Type', inplace=True)
    table[table=='n/e']=np.nan
    table = table.astype(float)
    table = table.transpose()
    table.to_parquet(output_path / (country + ".parquet"))

    cap_tables.append(table)