# %% [markdown]
# # Heart Disease Data Exploration
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
heart_data = pd.read_csv("HeartDisease/data/heart.csv")
# data retrieved from: https://www.kaggle.com/ronitf/heart-disease-uci
print(heart_data.shape)
heart_data.head()

# %% Clean data
# No missing/inf values
print("Missing:\n", heart_data.isna().sum(), "\n")
print("Inf:\n", (np.abs(heart_data) == np.inf).sum(), "\n")

heart_data.describe()



# %% Correlation plots
sns.pairplot(heart_data)
# plt.savefig('TFLCycles/images/pairplot.png')
plt.show()

# %% Interactive plots
# hvplot on dataframes
# panels/holoviews for interactivity
