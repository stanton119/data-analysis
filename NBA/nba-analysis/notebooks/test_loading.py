# %%
import pandas as pd

# %%
df = pd.read_parquet("../data/01_raw/seasons/2020.parquet")
df
# %%
df = pd.read_parquet("../data/02_intermediate/seasons.parquet")
df
# %%
df = pd.read_parquet("../data/07_model_output/shooting_dist.parquet")
df
# %%
from nba_analysis.pipelines.data_science.nodes import BetaBinomial

[BetaBinomial.from_state(df.iloc[idx,1]).plot_pdf() for idx in range(10)]
# %%
from kedro.io.data_catalog import DataCatalog
DataCatalog()
# %%
df[['FT','FTA']] * df['G']

(df['FT'] * df['G']).round()
df['FT_Scored']= df['FT'] * df['G']
df['FT_Attempted'] = df['FTA'] * df['G']

# %%
df[['Player','Age']].value_counts()>

# %%
df[['FT','FTA']] * df['G']
df['FTA'] * df['G']
# %%
df.dtypes
# %%
