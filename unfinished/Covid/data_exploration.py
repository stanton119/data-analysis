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
