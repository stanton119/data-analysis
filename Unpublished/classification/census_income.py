# %%
import pandas as pd

# %%
"""
Data descriptions available:
'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names'
"""
columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

df_train = pd.read_csv(
    r"https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    header=None,
    sep=', ',
    na_values=['?'],
)
df_train.columns = columns

df_test = pd.read_csv(
    r"https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    header=None,
    skiprows=[0],
    sep=', ',
    na_values=['?'],
)
df_test.columns = columns

# %%
# clean data
# filter NaNs, assuming randomly distributed (?)
print(f"Removing no. rows: {df_train.isna().any(axis=1).sum()}")
df_train = df_train.loc[df_train.notna().all(axis=1)]


# %%
import lightgbm as lgb
# %%
